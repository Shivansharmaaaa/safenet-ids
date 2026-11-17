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
    reasons = {(r.proto, r.dst_ip): out_of_scope(r) for r in t.expire(0, flush_all=True)}
    assert "multicast" in reasons[("udp", "224.0.0.251")]
    assert "icmp" in reasons[("icmp", "10.0.0.1")]
    assert "discovery" in reasons[("udp", "10.0.0.9")]
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
