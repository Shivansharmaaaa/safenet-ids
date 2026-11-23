"""Unit tests for the parts of SafeNet that must behave identically on datasets and live traffic.
Packets are built with scapy so the flow meter is exercised exactly as on a real capture."""
from __future__ import annotations

import ipaddress

import numpy as np
import pandas as pd
import pytest
from scapy.layers.inet import ICMP, IP, TCP, UDP

from safenet import adversarial as adv
from safenet.behavior import SCAN_PORTS, BehaviorDetector
from safenet.features import COMMON_FEATURES, from_cicids2017, from_unswnb15, service_from_port
from safenet.flowmeter import FlowTable, _wire_segments
from safenet.intel import ThreatFeeds, mitre_for, remote_ip
from safenet.policy import out_of_scope


def _pkt(p, t):
    p = IP(bytes(p))  # serialise -> parse, exactly like a packet read from the wire (fills len/checksums)
    p.time = t
    return p


def tcp_session(t0=1000.0, client="10.0.0.2", server="93.184.216.34", sport=51000, dport=443):
    return [
        _pkt(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="S", window=64240), t0),
        _pkt(IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="SA", window=65535), t0 + 0.05),
        _pkt(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="A", window=501) / (b"x" * 100), t0 + 0.10),
        _pkt(IP(src=server, dst=client) / TCP(sport=dport, dport=sport, flags="A", window=501) / (b"y" * 900), t0 + 0.15),
        _pkt(IP(src=client, dst=server) / TCP(sport=sport, dport=dport, flags="FA", window=501), t0 + 0.20),
    ]


# ------------------------------------------------------------------ flow meter
def test_flowmeter_bidirectional_features():
    t = FlowTable()
    for p in tcp_session():
        t.on_packet(p)
    (rec,) = t.expire(0, flush_all=True)
    f = rec.features()
    assert list(f) == COMMON_FEATURES
    assert (rec.src_ip, rec.dst_port) == ("10.0.0.2", 443)
    assert f["fwd_pkts"] == 3 and f["bwd_pkts"] == 2
    assert f["fwd_bytes"] == 40 + 140 + 40 and f["bwd_bytes"] == 40 + 940
    assert f["fwd_win"] == 64240 and f["bwd_win"] == 65535
    assert f["duration_s"] == pytest.approx(0.20)
    assert f["svc_tls"] == 1 and f["is_tcp"] == 1


def test_flowmeter_orients_mid_stream_capture_client_to_server():
    t = FlowTable()
    # first packet seen comes FROM the server
    t.on_packet(_pkt(IP(src="93.184.216.34", dst="10.0.0.2") / TCP(sport=443, dport=51000, flags="A"), 5.0))
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="93.184.216.34") / TCP(sport=51000, dport=443, flags="A"), 5.1))
    (rec,) = t.expire(0, flush_all=True)
    assert rec.src_ip == "10.0.0.2" and rec.dst_port == 443
    assert rec.bwd_pkts == 1 and rec.fwd_pkts == 1


def test_orientation_uses_handshake_and_ephemeral_ports():
    t = FlowTable()
    # mid-stream Apple push connection: first packet seen is server 5223 -> client ephemeral port
    t.on_packet(_pkt(IP(src="17.242.13.4", dst="10.0.0.2") / TCP(sport=5223, dport=57450, flags="PA") / b"z", 1.0))
    # a SYN from source port 53 (a classic firewall-evasion trick) is still the client
    t.on_packet(_pkt(IP(src="10.0.0.66", dst="10.0.0.2") / TCP(sport=53, dport=8080, flags="S"), 1.0))
    # first packet seen is the SYN-ACK: its sender is the server
    t.on_packet(_pkt(IP(src="10.0.0.9", dst="10.0.0.2") / TCP(sport=40000, dport=41000, flags="SA"), 1.0))
    recs = {r.dst_port: r for r in t.expire(0, flush_all=True)}
    assert recs[5223].src_ip == "10.0.0.2" and recs[5223].bwd_pkts == 1
    assert recs[8080].src_ip == "10.0.0.66" and recs[8080].src_port == 53
    assert recs[40000].src_ip == "10.0.0.2"


def test_tso_super_packets_are_resplit_to_wire_frames():
    assert _wire_segments(800, "tcp", False) == [800]
    segs = _wire_segments(4000, "tcp", False)
    assert all(s <= 1500 for s in segs) and sum(s - 40 for s in segs) == 4000 - 40
    assert _wire_segments(4000, "udp", False) == [4000]


def test_flow_expires_on_fin_and_idle():
    t = FlowTable()
    for p in tcp_session(t0=100.0):
        t.on_packet(p)
    assert t.expire(100.5) == []            # FIN seen but grace period not over
    assert len(t.expire(101.5)) == 1        # FIN + 1 s grace


# ------------------------------------------------------------------ dataset mappings
def test_cic_and_unsw_map_to_the_same_schema():
    cic = pd.DataFrame([{
        " Destination Port": 22, " Flow Duration": 2_000_000, " Total Fwd Packets": 10, " Total Backward Packets": 8,
        "Total Length of Fwd Packets": 1000, " Total Length of Bwd Packets": 2000, " Fwd Header Length": 320,
        " Bwd Header Length": 256, " Fwd IAT Mean": 200_000, " Bwd IAT Mean": 250_000,
        "Init_Win_bytes_forward": 29200, " Init_Win_bytes_backward": -1, "FIN Flag Count": 1, " SYN Flag Count": 1,
        " RST Flag Count": 0, " PSH Flag Count": 1, " ACK Flag Count": 1}])
    unsw = pd.DataFrame([{"dur": 2.0, "spkts": 10, "dpkts": 8, "sbytes": 1320, "dbytes": 2256, "sinpkt": 200.0,
                          "dinpkt": 250.0, "swin": 255, "dwin": 255, "proto": "tcp", "service": "ssh"}])
    a, b = from_cicids2017(cic), from_unswnb15(unsw)
    assert list(a.columns) == list(b.columns) == COMMON_FEATURES
    for col in ("duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "fwd_iat_ms", "svc_ssh", "is_tcp"):
        assert a[col].iloc[0] == pytest.approx(b[col].iloc[0]), col
    assert a["bwd_win"].iloc[0] == 0  # CIC's -1 ("not observed") becomes 0


def test_service_from_port():
    assert service_from_port(443) == "svc_tls" and service_from_port(53) == "svc_dns"
    assert service_from_port(31337) == "svc_other"


# ------------------------------------------------------------------ adversarial perturbations
def test_perturbations_respect_attacker_threat_model():
    X = pd.DataFrame([{c: 0.0 for c in COMMON_FEATURES}])
    X.loc[0, ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "fwd_pkt_mean", "bwd_pkt_mean",
              "fwd_iat_ms", "bwd_iat_ms"]] = [1.0, 10, 5, 1000, 500, 100, 100, 100, 200]
    P = adv.perturb(X, pad=1.0, slow=1.0, inject=0.5)
    r = P.iloc[0]
    assert r["bwd_bytes"] == 500 and r["bwd_pkts"] == 5          # victim side untouched
    assert r["fwd_pkt_mean"] <= adv.MTU
    assert r["duration_s"] == pytest.approx(2.0)
    assert r["fwd_pkts"] == 15
    assert r["bytes_per_s"] == pytest.approx((r["fwd_bytes"] + r["bwd_bytes"]) / r["duration_s"])


def test_adaptive_attack_never_increases_score():
    X = pd.DataFrame(np.abs(np.random.default_rng(0).normal(100, 30, (50, len(COMMON_FEATURES)))), columns=COMMON_FEATURES)
    score = lambda D: 1 / (1 + np.exp(-(D["fwd_pkt_mean"].values - 150) / 20))
    base = score(X)
    _, best = adv.adaptive_attack(X, score, eps=1.0, steps=3)
    assert (best <= base + 1e-12).all()


# ------------------------------------------------------------------ behaviour + policy + intel
def test_port_scan_rule_fires_once():
    t = FlowTable()
    for i, port in enumerate(range(1, SCAN_PORTS + 5)):
        ts = 200.0 + i * 0.01
        t.on_packet(_pkt(IP(src="10.0.0.2", dst="45.33.32.156") / TCP(sport=40000 + i, dport=port, flags="S"), ts))
        t.on_packet(_pkt(IP(src="45.33.32.156", dst="10.0.0.2") / TCP(sport=port, dport=40000 + i, flags="RA"), ts + .1))
    recs = t.expire(0, flush_all=True)
    det = BehaviorDetector()
    alerts = det.update(recs, list(range(len(recs))), [{"p_attack": 0.0}] * len(recs))
    assert len(alerts) == 1 and alerts[0][1]["label"] == "Scan"
    assert alerts[0][1]["explanation"]["evidence"]["distinct_ports"] >= SCAN_PORTS
    assert det.update(recs, list(range(len(recs))), [{"p_attack": 0.0}] * len(recs)) == []  # cooldown


def test_out_of_scope_policy():
    t = FlowTable()
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="224.0.0.251") / UDP(sport=5353, dport=5353), 1.0))
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="10.0.0.1") / ICMP(), 1.0))
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="10.0.0.9") / UDP(sport=5353, dport=5353), 1.0))
    for p in tcp_session():
        t.on_packet(p)
    # DNS: one query left unanswered, one answered
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="10.0.0.53") / UDP(sport=50001, dport=53) / (b"q" * 28), 1.0))
    t.on_packet(_pkt(IP(src="10.0.0.2", dst="1.1.1.1") / UDP(sport=50002, dport=53) / (b"q" * 28), 1.0))
    t.on_packet(_pkt(IP(src="1.1.1.1", dst="10.0.0.2") / UDP(sport=53, dport=50002) / (b"a" * 60), 1.02))
    # QUIC packets from a server whose client side was never captured
    t.on_packet(_pkt(IP(src="17.253.18.81", dst="10.0.0.2") / UDP(sport=443, dport=61000) / (b"s" * 90), 1.0))
    reasons = {(r.proto, r.dst_ip): out_of_scope(r) for r in t.expire(0, flush_all=True)}
    assert "multicast" in reasons[("udp", "224.0.0.251")]
    assert "icmp" in reasons[("icmp", "10.0.0.1")]
    assert "discovery" in reasons[("udp", "10.0.0.9")]
    assert "unanswered DNS" in reasons[("udp", "10.0.0.53")]
    assert reasons[("udp", "1.1.1.1")] is None
    assert "one-sided" in reasons[("udp", "17.253.18.81")]
    assert reasons[("tcp", "93.184.216.34")] is None


def test_blocklist_matching_is_local():
    f = ThreatFeeds()
    f.c2_ips = {"198.51.100.7"}
    f.drop_nets = [ipaddress.ip_network("203.0.113.0/24")]
    assert f.match("198.51.100.7")[0]["feed"] == "feodo"
    assert f.match("203.0.113.50")[0]["feed"] == "spamhaus_drop"
    assert f.match("8.8.8.8") == []


def test_remote_ip_and_mitre():
    assert remote_ip("10.0.0.2", "8.8.8.8") == "8.8.8.8"
    assert remote_ip("8.8.4.4", "192.168.1.5") == "8.8.4.4"
    assert remote_ip("10.0.0.2", "10.0.0.3") is None
    assert [m["id"] for m in mitre_for("Scan")] == ["T1046", "T1595"]
    assert mitre_for("Anomaly") == []


# ------------------------------------------------------------------ per-host sequence model
def _seq_flows():
    rng = np.random.default_rng(1)
    rows = []
    for host, n in (("10.0.0.2", 5), ("10.0.0.3", 3)):
        for k in range(n):
            rows.append(dict(host=host, ts=1000.0 + 2.6 * k + (0.3 if host == "10.0.0.3" else 0.0),
                             dst="93.184.216.34" if k % 2 else "1.1.1.1", dport=443 if k < 3 else 22,
                             **{c: float(rng.integers(0, 3000)) for c in COMMON_FEATURES}))
    return pd.DataFrame(rows)


def test_sequence_windows_match_between_training_and_live():
    from safenet import sequence as sq
    df = _seq_flows()                       # already sorted by (host, ts)
    steps = sq.step_matrix(df, df.host.values, df.ts.values, df.dst.values, df.dport.values)
    start = sq.host_starts(df.host.values)
    win_train = sq.gather(steps, start, np.arange(len(df)))
    hist = sq.HostHistory()
    for i, r in df.iterrows():
        win_live, metas = hist.push(r.host, r.ts, r.dst, r.dport, r[COMMON_FEATURES].values.astype(np.float32), {"i": i})
        np.testing.assert_allclose(win_live, win_train[i], rtol=1e-5, atol=1e-5)
    present = sq.STEP_FEATURES.index("present")
    assert win_train[2][:, present].tolist()[-4:] == [0, 1, 1, 1]          # 3rd flow of host A: 3 steps present
    assert win_train[5][:, present].sum() == 1                             # 1st flow of host B: no context leaks in
    gap = sq.STEP_FEATURES.index("gap_s")
    assert np.expm1(win_train[1][-1, gap]) == pytest.approx(2.0)           # floor(1002.6) - floor(1000.0) whole seconds


def test_sequence_lstm_forward():
    import torch
    from safenet import sequence as sq
    m = sq.SeqLSTM()
    out = m(torch.zeros(4, sq.SEQ_LEN, len(sq.STEP_FEATURES)))
    assert out.shape == (4,)


def test_lstm_alerts_from_one_host_are_grouped_into_one_campaign():
    from types import SimpleNamespace
    from scripts.sensor import CAMPAIGN_GAP_S, Scorer
    t = FlowTable()
    for i, port in enumerate(range(1, 31)):
        ts = 500.0 + i * 0.05
        t.on_packet(_pkt(IP(src="10.0.0.2", dst="45.33.32.156") / TCP(sport=41000 + i, dport=port, flags="S"), ts))
        t.on_packet(_pkt(IP(src="45.33.32.156", dst="10.0.0.2") / TCP(sport=port, dport=41000 + i, flags="RA"), ts + .1))
    recs = sorted(t.expire(0, flush_all=True), key=lambda r: r.first_ts)
    s = object.__new__(Scorer)                                  # no model files needed: stub the LSTM
    s.campaigns = {}
    s.seq = SimpleNamespace(thr=0.9, score=lambda rs, p: {i: (0.99, None, []) for i in range(len(rs))},
                            explainer=SimpleNamespace(explain=lambda w, m: {"detector": "sequence", "context_flows": 15,
                                                                             "contributions": []}))
    out = [dict(p_attack=0.0, is_alert=False) for _ in recs]
    s._sequence(recs, out, ["Scan"] * len(recs), np.zeros(len(recs)))
    assert sum("alert" in o for o in out) == 1 and all(o["is_alert"] for o in out)
    (fid, camp), = Scorer.campaign_updates(list(range(100, 100 + len(out))), out)
    assert fid == 100 and camp["flows"] == 30 and camp["distinct_ports"] == 30 and camp["distinct_hosts"] == 1
    # the same host much later is a new incident
    late = recs[0]
    late.first_ts += CAMPAIGN_GAP_S + 1
    out2 = [dict(p_attack=0.0, is_alert=False)]
    s._sequence([late], out2, ["Scan"], np.zeros(1))
    assert "alert" in out2[0]


def test_adaptive_adversarial_training_only_perturbs_attacker_flows():
    import torch
    from safenet import sequence as sq
    from scripts.train_sequence import adaptive_augmenter
    rng = np.random.default_rng(0)
    n = 200
    raw = np.abs(rng.normal(5, 3, (n, len(COMMON_FEATURES)))).astype(np.float32)
    host, ts = np.repeat(np.arange(10), 20), np.tile(np.arange(20) * 3.0, 10)
    dst, dport = rng.integers(0, 3, n), rng.integers(0, 3, n)
    steps = sq.step_matrix(pd.DataFrame(raw, columns=COMMON_FEATURES), host, ts, dst, dport)
    start = sq.host_starts(host)
    y = (rng.random(n) < 0.5).astype(np.int8)
    idx = np.arange(n)
    base = sq.gather(steps, start, idx)
    xb = base.copy()
    torch.manual_seed(0)
    adaptive_augmenter(raw, y, start)(xb, idx, rng, sq.SeqLSTM(drop=sq.TIMING_FEATURES).eval())
    changed = np.abs(xb - base).sum(-1) > 1e-6                       # [window, step]
    offs = np.arange(sq.SEQ_LEN - 1, -1, -1)
    rows = idx[:, None] - offs[None, :]
    valid = rows >= start[idx][:, None]
    attacker = (y[np.where(valid, rows, 0)] == 1) & valid
    assert changed.any() and not (changed & ~attacker).any()          # benign flows and padding untouched
    assert (y[changed.any(1)] == 1).all()                             # only attack windows are rewritten


def test_watchdog_exits_a_stalled_sensor():
    import subprocess
    import sys
    code = ("import time, threading, scripts.sensor as s\n"
            "s.WATCHDOG_S = 1\ns.WATCHDOG_CHECK_S = 0.2\ns.progress()\n"
            "threading.Thread(target=s._watchdog, daemon=True).start()\ntime.sleep(60)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=50)
    assert r.returncode == 3 and "watchdog" in r.stdout
