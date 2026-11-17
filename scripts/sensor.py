"""SafeNet live sensor: capture real packets -> flows -> ML scoring -> SHAP -> database.

usage (needs root for packet capture on macOS/Linux):
    sudo .venv/bin/python -m scripts.sensor --iface en0
    .venv/bin/python -m scripts.sensor --pcap capture.pcap     # score a real capture file
"""
from __future__ import annotations

import argparse
import signal
import socket
import sys
import time
from urllib.parse import urlparse

import joblib
import numpy as np

from safenet import db
from safenet.config import MODELS_DIR, ROOT, database_url
from safenet.explain import AlertExplainer
from safenet.features import to_model_space
from safenet.flowmeter import FlowRecord, FlowTable

STOP = False


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Scorer:
    def __init__(self, path=MODELS_DIR / "live_bundle.joblib"):
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
            is_alert = p >= self.thr or strong_anomaly
            label = cat_lab[i] if p >= 0.5 else "Benign"
            rec = dict(
                ts_start=r.first_ts, ts_end=r.last_ts, src_ip=r.src_ip, src_port=r.src_port, dst_ip=r.dst_ip,
                dst_port=r.dst_port, proto=r.proto, features=dict(zip(self.features, raw[i].tolist())),
                pred_label=label, p_attack=p, ae_score=float(ae_s[i]), ae_anomaly=anomaly, is_alert=is_alert,
            )
            if is_alert:
                expl = self.explainer.explain(X[i], raw[i])
                alert_label = cat_lab[i] if p >= self.thr else "Anomaly"
                severity = "high" if (p >= 0.98 or (p >= self.thr and anomaly)) else ("medium" if p >= self.thr else "low")
                expl["ae_score"] = float(ae_s[i])
                expl["ae_threshold"] = float(self.ae.threshold)
                rec["alert"] = dict(
                    severity=severity, label=alert_label, p_attack=p, ae_score=float(ae_s[i]), src_ip=r.src_ip,
                    dst_ip=r.dst_ip, dst_port=r.dst_port, proto=r.proto, explanation=expl,
                    summary=AlertExplainer.summarise(alert_label, expl),
                )
            out.append(rec)
        return out


def db_host_ips(url: str) -> set[str]:
    host = urlparse(url).hostname
    ips = set()
    try:
        for info in socket.getaddrinfo(host, None):
            ips.add(info[4][0])
    except OSError:
        pass
    return ips


def flush(conn, sensor_id: int, scorer: Scorer, recs: list[FlowRecord], stats: dict) -> None:
    if not recs:
        return
    scored = scorer.score(recs)
    n_alert = 0
    for rec in scored:
        flow_id = db.insert_flow(conn, sensor_id, rec)
        if "alert" in rec:
            db.insert_alert(conn, flow_id, rec["alert"])
            n_alert += 1
            log(f"ALERT {rec['alert']['severity']:6s} {rec['alert']['label']:10s} p={rec['p_attack']:.3f} "
                f"{rec['src_ip']}:{rec['src_port']} -> {rec['dst_ip']}:{rec['dst_port']}/{rec['proto']}")
    stats["flows"] += len(scored)
    db.heartbeat(conn, sensor_id, stats["packets"], stats["flows"])
    conn.commit()
    log(f"scored {len(scored)} flows ({n_alert} alerts); totals packets={stats['packets']} flows={stats['flows']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=None, help="interface to sniff (default: system default route)")
    ap.add_argument("--pcap", default=None, help="score a capture file instead of live traffic")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    from scapy.all import AsyncSniffer, conf, sniff

    url = database_url()
    db.init_schema(url)
    scorer = Scorer()
    iface = args.iface or (f"pcap:{args.pcap}" if args.pcap else str(conf.iface))
    table = FlowTable(ignore_ips=db_host_ips(url))
    stats = {"packets": 0, "flows": 0}
    log(f"SafeNet sensor model={scorer.version} iface={iface} ignoring DB hosts {sorted(table.ignore)}")

    with db.connect(url) as conn:
        sensor_id = db.upsert_sensor(conn, socket.gethostname(), iface, scorer.version)

        if args.pcap:
            sniff(offline=args.pcap, prn=table.on_packet, store=False)
            stats["packets"] = table.packets
            flush(conn, sensor_id, scorer, table.expire(0, flush_all=True), stats)
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
        # When the model bundle or .env (database URL) changes, exit cleanly; launchd
        # (KeepAlive) restarts the sensor with the new model / database.
        watched = [MODELS_DIR / "live_bundle.joblib", ROOT / ".env"]
        mtimes = {p: p.stat().st_mtime if p.exists() else 0 for p in watched}
        try:
            while not STOP:
                time.sleep(args.interval)
                stats["packets"] = table.packets
                try:
                    flush(conn, sensor_id, scorer, table.expire(time.time()), stats)
                    if time.time() - last_prune > 3600:
                        db.prune(conn, keep_days=3)
                        conn.commit()
                        last_prune = time.time()
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
                    log(f"{changed} changed; restarting sensor to pick up new model/config")
                    break
        finally:
            sniffer.stop()
            stats["packets"] = table.packets
            flush(conn, sensor_id, scorer, table.expire(time.time(), flush_all=True), stats)
            log("sensor stopped")


if __name__ == "__main__":
    sys.exit(main())
