"""SafeNet live sensor: capture real packets -> flows -> ML scoring -> SHAP -> threat intel -> database.

usage (needs root for packet capture on macOS/Linux):
    sudo .venv/bin/python -m scripts.sensor --iface en0
    .venv/bin/python -m scripts.sensor --pcap capture.pcap     # score a real capture file
"""
from __future__ import annotations

import argparse
import queue
import signal
import socket
import sys
import threading
import time
from urllib.parse import urlparse

import joblib
import numpy as np

from safenet import db
from safenet.behavior import BehaviorDetector
from safenet.config import MODELS_DIR, ROOT, database_url
from safenet.explain import AlertExplainer
from safenet.features import to_model_space
from safenet.flowmeter import FlowRecord, FlowTable
from safenet.policy import out_of_scope
from safenet.intel import ThreatFeeds, enrich_ip, mitre_for, remote_ip
from safenet.sequence import SEQ_LEN, STEP_FEATURES, HostHistory, SeqExplainer, SeqLSTM

STOP = False
# A host's LSTM alerts within CAMPAIGN_GAP_S of each other form one campaign: one alert, with the later flows folded
# into it (a 50-port scan is one incident, not 50). A campaign is closed after CAMPAIGN_MAX_S so it cannot run forever.
CAMPAIGN_GAP_S = 600.0
CAMPAIGN_MAX_S = 3600.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class SeqScorer:
    """Per-host LSTM: scores every TCP/UDP flow in the context of the same host's previous flows."""

    def __init__(self, path=MODELS_DIR / "sequence_bundle.pt"):
        import torch
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b["step_features"] != STEP_FEATURES or b["seq_len"] != SEQ_LEN:
            raise ValueError("sequence bundle was trained with a different step-feature layout")
        self.version, self.thr = b["version"], float(b["alert_threshold"])
        self.model = SeqLSTM(b["n_feat"], drop=b.get("drop", []))
        state = dict(b["state_dict"])
        state.setdefault("mask", self.model.mask)          # bundles from before feature masking: keep everything
        self.model.load_state_dict(state)
        self.model.eval()
        self.hist = HostHistory(b["seq_len"])
        self.explainer = SeqExplainer(self.model, b["background"])

    def score(self, recs: list[FlowRecord], p_flow: list[float]) -> dict[int, tuple[float, np.ndarray, list]]:
        """{index in recs: (p_seq, window, step metadata)}; flows enter each host's history in start order."""
        import torch
        order = sorted(range(len(recs)), key=lambda i: recs[i].first_ts)
        idx, wins, metas = [], [], []
        for i in order:
            r = recs[i]
            if r.proto not in ("tcp", "udp"):
                continue
            meta = {"ts": r.first_ts, "dst": r.dst_ip, "dport": r.dst_port, "proto": r.proto,
                    "p_flow": round(float(p_flow[i]), 4)}
            w, m = self.hist.push(r.src_ip, r.first_ts, r.dst_ip, r.dst_port, r.vector(), meta)
            idx.append(i); wins.append(w); metas.append(m)
        if not idx:
            return {}
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(np.stack(wins)))).numpy()
        self.hist.prune(time.time())
        return {i: (float(p[k]), wins[k], metas[k]) for k, i in enumerate(idx)}


class Scorer:
    def __init__(self, feeds: ThreatFeeds | None = None, path=MODELS_DIR / "live_bundle.joblib"):
        b = joblib.load(path)
        self.version = b["version"]
        self.features = b["features"]
        self.det = b["detector"]
        self.cat = b["category_model"]
        self.cats = b["categories"]
        self.ae = b["autoencoder"]
        self.thr = float(b["alert_threshold"])
        self.explainer = AlertExplainer(self.det, self.features)
        self.benign_i = self.cats.index("Benign")
        self.feeds = feeds
        seq_path = MODELS_DIR / "sequence_bundle.pt"
        self.seq = SeqScorer(seq_path) if seq_path.exists() else None
        self.campaigns: dict[str, dict] = {}      # source host -> its open LSTM campaign

    def score(self, recs: list[FlowRecord]) -> list[dict]:
        raw = np.vstack([r.vector() for r in recs])
        X = to_model_space(raw)
        p_att = self.det.predict_proba(X)[:, 1]
        cat_p = self.cat.predict_proba(X)
        cat_p[:, self.benign_i] = -1  # pick the most likely ATTACK type for labelling
        cat_lab = [self.cats[i] for i in cat_p.argmax(1)]
        ae_s = self.ae.score(X)
        out = []
        for i, r in enumerate(recs):
            p = float(p_att[i])
            anomaly = bool(ae_s[i] > self.ae.threshold)
            strong_anomaly = bool(ae_s[i] > 5 * self.ae.threshold and (r.fwd_pkts + r.bwd_pkts) >= 3)
            rip = remote_ip(r.src_ip, r.dst_ip)
            hits = self.feeds.match(rip) if (self.feeds and rip) else []
            model_alert = p >= self.thr or strong_anomaly
            scope = out_of_scope(r)
            suppressed = scope if (model_alert and scope) else None
            is_alert = (model_alert and not scope) or bool(hits)
            label = cat_lab[i] if p >= 0.5 else "Benign"
            rec = dict(
                ts_start=r.first_ts, ts_end=r.last_ts, src_ip=r.src_ip, src_port=r.src_port, dst_ip=r.dst_ip,
                dst_port=r.dst_port, proto=r.proto, features=dict(zip(self.features, raw[i].tolist())),
                pred_label=label, p_attack=p, ae_score=float(ae_s[i]), ae_anomaly=anomaly, is_alert=is_alert,
                suppressed=suppressed,
            )
            if is_alert:
                expl = self.explainer.explain(X[i], raw[i])
                expl["ae_score"] = float(ae_s[i])
                expl["ae_threshold"] = float(self.ae.threshold)
                expl["detector"] = "threat-intel" if hits else ("ml" if p >= self.thr else "autoencoder")
                if hits:
                    alert_label, severity = "ThreatIntel", "high"
                    summary = (f"ThreatIntel: {rip} is listed on " + ", ".join(h["name"] for h in hits)
                               + f" (model attack score {p:.2f}).")
                else:
                    alert_label = cat_lab[i] if p >= self.thr else "Anomaly"
                    severity = ("high" if (p >= 0.98 or (p >= self.thr and anomaly))
                                else ("medium" if p >= self.thr else "low"))
                    summary = AlertExplainer.summarise(alert_label, expl)
                rec["alert"] = dict(
                    severity=severity, label=alert_label, p_attack=p, ae_score=float(ae_s[i]), src_ip=r.src_ip,
                    dst_ip=r.dst_ip, dst_port=r.dst_port, proto=r.proto, explanation=expl, summary=summary,
                    remote_ip=rip, mitre=mitre_for(alert_label), intel_hits=hits,
                )
            out.append(rec)
        if self.seq:
            self._sequence(recs, out, cat_lab, ae_s)
        return out

    def _sequence(self, recs, out, cat_lab, ae_s) -> None:
        for i, (p, win, metas) in self.seq.score(recs, [o["p_attack"] for o in out]).items():
            rec, r = out[i], recs[i]
            rec["p_seq"] = p
            if p < self.seq.thr:
                continue
            if "alert" in rec:                       # already alerted per flow: record the context score too
                rec["alert"]["explanation"]["p_seq"] = p
                continue
            scope = out_of_scope(r)
            if scope:
                rec["suppressed"] = rec.get("suppressed") or scope
                continue
            label = cat_lab[i]
            c = self.campaigns.get(r.src_ip)
            if c and r.first_ts - c["last"] <= CAMPAIGN_GAP_S and r.first_ts - c["first"] <= CAMPAIGN_MAX_S:
                c["last"] = max(c["last"], r.first_ts)
                c["n"] += 1
                c["max_p"] = max(c["max_p"], p)
                c["hosts"].add(r.dst_ip)
                c["ports"].add(r.dst_port)
                rec["is_alert"] = True                # part of an incident: kept as evidence, no second alert
                rec["seq_campaign"] = c
                continue
            c = self.campaigns[r.src_ip] = dict(first=r.first_ts, last=r.first_ts, n=1, max_p=p, hosts={r.dst_ip},
                                                ports={r.dst_port}, flow_id=None)
            rec["seq_campaign"] = c
            expl = self.seq.explainer.explain(win, metas)
            expl["p_seq"] = p
            n_ctx = expl["context_flows"]
            rip = remote_ip(r.src_ip, r.dst_ip)
            summary = (f"{label}: in the context of {r.src_ip}'s last {n_ctx} flows the LSTM scores this flow "
                       f"{p:.2f} (single-flow model {rec['p_attack']:.2f}). "
                       + AlertExplainer.summarise(label, expl).split(": ", 1)[1].capitalize())
            rec["is_alert"] = True
            rec["alert"] = dict(
                severity="high" if p >= 0.98 else "medium", label=label, p_attack=rec["p_attack"],
                ae_score=float(ae_s[i]), src_ip=r.src_ip, dst_ip=r.dst_ip, dst_port=r.dst_port, proto=r.proto,
                explanation=expl, summary=summary, remote_ip=rip, mitre=mitre_for(label), intel_hits=[],
            )
        horizon = time.time() - 2 * CAMPAIGN_MAX_S
        for host in [h for h, c in self.campaigns.items() if c["last"] < horizon]:
            del self.campaigns[host]

    @staticmethod
    def campaign_updates(ids: list[int], scored: list[dict]) -> list[tuple[int, dict]]:
        """After the flows are stored: (flow id of the campaign's alert, campaign summary) for every campaign that
        grew in this batch. The campaign's first flow carries the alert."""
        grown = {}
        for fid, rec in zip(ids, scored):
            c = rec.pop("seq_campaign", None)
            if c is None:
                continue
            if c["flow_id"] is None and "alert" in rec:
                c["flow_id"] = fid
            if c["n"] > 1:
                grown[id(c)] = c
        return [(c["flow_id"], dict(flows=c["n"], distinct_hosts=len(c["hosts"]), distinct_ports=len(c["ports"]),
                                    sample_ports=sorted(c["ports"])[:24], first_ts=c["first"], last_ts=c["last"],
                                    duration_s=round(c["last"] - c["first"], 1), max_p_seq=c["max_p"]))
                for c in grown.values() if c["flow_id"] is not None]


class Enricher(threading.Thread):
    """Background worker: looks up public IPs from alerts in ipwho.is / Shodan / GreyNoise."""

    def __init__(self, url: str, feeds: ThreatFeeds):
        super().__init__(daemon=True)
        self.url, self.feeds, self.q = url, feeds, queue.Queue()
        self.seen: dict[str, float] = {}

    def submit(self, ip: str | None) -> None:
        if ip and time.time() - self.seen.get(ip, 0) > 3600:
            self.seen[ip] = time.time()
            self.q.put(ip)

    def run(self) -> None:
        while True:
            ip = self.q.get()
            for attempt in range(1, 4):  # the network / database can be briefly unreachable
                try:
                    with db.connect(self.url) as conn:
                        if not db.ip_intel_fresh(conn, ip):
                            data = enrich_ip(ip, self.feeds.kev)
                            data["blocklists"] = self.feeds.match(ip)
                            db.save_ip_intel(conn, ip, data)
                            conn.commit()
                            log(f"intel {ip}: {data.get('geo', {}).get('country')} {data.get('geo', {}).get('org')} "
                                f"sources={data['sources']}")
                    break
                except Exception as e:
                    log(f"intel error for {ip} (attempt {attempt}/3): {str(e)[:160]!r}")
                    time.sleep(10 * attempt)
            time.sleep(1.5)  # stay well inside the free APIs' rate limits


def db_host_ips(url: str) -> set[str]:
    host = urlparse(url).hostname
    ips = set()
    try:
        for info in socket.getaddrinfo(host, None):
            ips.add(info[4][0])
    except OSError:
        pass
    return ips


BEHAVIOR = BehaviorDetector()


def flush(conn, sensor_id: int, scorer: Scorer, recs: list[FlowRecord], stats: dict,
          enricher: Enricher | None = None) -> None:
    if not recs:                                            # quiet network: still tell the dashboard we are alive
        db.heartbeat(conn, sensor_id, stats["packets"], stats["flows"])
        conn.commit()
        return
    scored = scorer.score(recs)
    ids = db.insert_flows(conn, sensor_id, scored)          # one round trip per 500 flows
    alerts = [(fid, rec["alert"]) for fid, rec in zip(ids, scored) if "alert" in rec]
    # multi-flow behavioural rules (port scans, sweeps, brute force)
    for fid, a in BEHAVIOR.update(recs, ids, scored):
        a["remote_ip"] = remote_ip(a["src_ip"], a["dst_ip"])
        a["mitre"] = mitre_for(a["label"])
        a["intel_hits"] = scorer.feeds.match(a["remote_ip"]) if (scorer.feeds and a["remote_ip"]) else []
        alerts.append((fid, a))
    if alerts:
        conn.execute("update safenet.flows set is_alert = true where id = any(%s)", ([f for f, _ in alerts],))
    db.insert_alerts(conn, alerts)
    for fid, camp in scorer.campaign_updates(ids, scored):
        db.update_campaign(conn, fid, camp)
        log(f"LSTM campaign {camp['flows']} flows, {camp['distinct_ports']} ports, {camp['distinct_hosts']} hosts "
            f"(alert on flow {fid})")
    n_alert = len(alerts)
    for _, a in alerts:
        if enricher:
            enricher.submit(a["remote_ip"])
        log(f"ALERT {a['severity']:6s} {a['label']:11s} p={a['p_attack']:.3f} "
            f"{a['src_ip']} -> {a['dst_ip']}:{a['dst_port']}/{a['proto']}")
    stats["flows"] += len(scored)
    db.heartbeat(conn, sensor_id, stats["packets"], stats["flows"])
    conn.commit()
    log(f"scored {len(scored)} flows ({n_alert} alerts); totals packets={stats['packets']} flows={stats['flows']}")


def refresh_feeds(url: str, feeds: ThreatFeeds, force: bool = False) -> None:
    status = feeds.refresh(force=force)
    with db.connect(url) as c:
        db.save_feed_status(c, status)
        c.commit()
    log("threat feeds: " + ", ".join(f"{k}={v['entries']}{'' if v['ok'] else '(stale)'}" for k, v in status.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None, help="interface to sniff (default: system default route)")
    ap.add_argument("--pcap", default=None, help="score a capture file instead of live traffic")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    from scapy.all import AsyncSniffer, conf, sniff

    url = database_url()
    db.init_schema(url)
    feeds = ThreatFeeds()
    refresh_feeds(url, feeds, force=True)
    scorer = Scorer(feeds)
    iface = args.iface or (f"pcap:{args.pcap}" if args.pcap else str(conf.iface))
    table = FlowTable(ignore_ips=db_host_ips(url))
    stats = {"packets": 0, "flows": 0}
    enricher = Enricher(url, feeds)
    enricher.start()
    log(f"SafeNet sensor model={scorer.version} sequence={scorer.seq.version if scorer.seq else 'none'} "
        f"iface={iface} ignoring DB hosts {sorted(table.ignore)}")

    with db.connect(url) as conn:
        version = scorer.version + (f" + {scorer.seq.version}" if scorer.seq else "")
        sensor_id = db.upsert_sensor(conn, socket.gethostname(), iface, version)

        if args.pcap:
            sniff(offline=args.pcap, prn=table.on_packet, store=False)
            stats["packets"] = table.packets
            flush(conn, sensor_id, scorer, table.expire(0, flush_all=True), stats, enricher)
            deadline = time.time() + 60          # let the intel lookups for these alerts finish
            while not enricher.q.empty() and time.time() < deadline:
                time.sleep(0.5)
            time.sleep(5)
            return

        def _stop(*_):
            global STOP
            STOP = True
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        sniffer = AsyncSniffer(iface=args.iface or None, prn=table.on_packet, store=False, filter="ip or ip6")
        sniffer.start()
        last_prune = 0.0
        errors = 0
        # When the model, .env (database URL) or the sensor code changes, exit cleanly;
        # launchd (KeepAlive) restarts the sensor with the new version.
        watched = [MODELS_DIR / "live_bundle.joblib", MODELS_DIR / "sequence_bundle.pt", ROOT / ".env", ROOT / "scripts" / "sensor.py",
                   *sorted((ROOT / "safenet").glob("*.py"))]
        mtimes = {p: p.stat().st_mtime if p.exists() else 0 for p in watched}
        try:
            while not STOP:
                time.sleep(args.interval)
                stats["packets"] = table.packets
                try:
                    flush(conn, sensor_id, scorer, table.expire(time.time()), stats, enricher)
                    if time.time() - last_prune > 3600:
                        db.prune(conn, keep_days=3)
                        conn.commit()
                        last_prune = time.time()
                    if time.time() - feeds.loaded_at > feeds.ttl:
                        threading.Thread(target=refresh_feeds, args=(url, feeds), daemon=True).start()
                        feeds.loaded_at = time.time()  # avoid starting a second refresh meanwhile
                    errors = 0
                except Exception as e:  # keep sensing even if one DB write fails
                    errors += 1
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    log(f"db error ({errors}): {e!r}")
                    if errors >= 5:
                        log("database unreachable; exiting so the service restarts with a fresh connection")
                        break
                changed = [p.name for p in watched if (p.stat().st_mtime if p.exists() else 0) != mtimes[p]]
                if changed:
                    log(f"{changed} changed; restarting sensor to pick up new model/config/code")
                    break
                if not sniffer.running or (sniffer.thread is not None and not sniffer.thread.is_alive()):
                    # scapy closes a socket whose callback raised; never keep heartbeating without capturing
                    log("packet capture stopped unexpectedly; exiting so the service restarts it")
                    break
        finally:
            try:
                sniffer.stop()
            except Exception:
                pass
            stats["packets"] = table.packets
            try:
                flush(conn, sensor_id, scorer, table.expire(time.time(), flush_all=True), stats, enricher)
            except Exception as e:
                log(f"final flush failed: {e!r}")
            log("sensor stopped")


if __name__ == "__main__":
    sys.exit(main())
