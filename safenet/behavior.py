"""Behavioural (multi-flow) detection that complements the per-flow ML models.

A single probe of a port scan or a single failed login looks like an ordinary short
connection, so per-flow classifiers trained on one network often miss them on another
(SafeNet measured this live: a TCP connect scan from macOS over the internet scored
benign flow-by-flow). Like Zeek's scan policy or Suricata thresholds, these rules
aggregate flows from the same source over a sliding window:

  * vertical port scan   one source -> one host, >= SCAN_PORTS distinct ports
  * horizontal sweep     one source -> >= SWEEP_HOSTS hosts on the same port
  * brute force          one source -> same host:auth-port, >= BRUTE_CONN short connections
Only "probe-like" flows count: no payload came back (failed/refused/unanswered connections),
so normal browsing to many sites does not trigger the sweep rule.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from .flowmeter import FlowRecord

WINDOW_S = 60.0
SCAN_PORTS = 25
SWEEP_HOSTS = 25
BRUTE_CONN = 15
AUTH_PORTS = {21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 110: "POP3", 143: "IMAP", 445: "SMB",
              1433: "MSSQL", 3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL", 5900: "VNC"}
COOLDOWN_S = 600.0


def _probe_like(r: FlowRecord) -> bool:
    """Connection attempt that got nothing useful back (closed/filtered port or refused)."""
    return r.proto == "tcp" and r.fwd_pkts <= 3 and r.bwd_bytes <= 120


def _short_session(r: FlowRecord) -> bool:
    return r.proto == "tcp" and (r.fwd_pkts + r.bwd_pkts) <= 40 and (r.last_ts - r.first_ts) <= 15


class BehaviorDetector:
    def __init__(self):
        self.events: deque[tuple[float, FlowRecord, int]] = deque()
        self.last_fired: dict[tuple, float] = {}

    def _cool(self, key: tuple, now: float) -> bool:
        if now - self.last_fired.get(key, 0) < COOLDOWN_S:
            return False
        self.last_fired[key] = now
        return True

    def update(self, recs: list[FlowRecord], flow_ids: list[int], scores: list[dict]) -> list[tuple[int, dict]]:
        now = time.time()
        for r, fid in zip(recs, flow_ids):
            self.events.append((r.last_ts, r, fid))
        horizon = max((e[0] for e in self.events), default=now) - WINDOW_S
        while self.events and self.events[0][0] < horizon:
            self.events.popleft()
        max_p = {fid: s["p_attack"] for fid, s in zip(flow_ids, scores)}

        ports = defaultdict(set)        # (src, dst) -> ports probed
        hosts = defaultdict(set)        # (src, port) -> hosts probed
        brute = defaultdict(list)       # (src, dst, port) -> short sessions
        last = {}
        for ts, r, fid in self.events:
            if _probe_like(r):
                ports[(r.src_ip, r.dst_ip)].add(r.dst_port)
                hosts[(r.src_ip, r.dst_port)].add(r.dst_ip)
            if r.dst_port in AUTH_PORTS and _short_session(r):
                brute[(r.src_ip, r.dst_ip, r.dst_port)].append(r)
            last[(r.src_ip, r.dst_ip)] = (fid, r)

        out = []
        for (src, dst), ps in ports.items():
            if len(ps) >= SCAN_PORTS and self._cool(("scan", src, dst), now):
                fid, r = last[(src, dst)]
                out.append((fid, self._alert("Scan", "vertical port scan", src, dst, r, max_p.get(fid, 0.0), {
                    "distinct_ports": len(ps), "window_s": WINDOW_S, "threshold": SCAN_PORTS,
                    "sample_ports": sorted(ps)[:25]},
                    f"Scan: {src} probed {len(ps)} distinct ports on {dst} within {int(WINDOW_S)} s "
                    f"(behavioural rule, threshold {SCAN_PORTS}).")))
        for (src, port), hs in hosts.items():
            if len(hs) >= SWEEP_HOSTS and self._cool(("sweep", src, port), now):
                dst = sorted(hs)[0]
                fid, r = last.get((src, dst), (None, None))
                if fid is None:
                    continue
                out.append((fid, self._alert("Scan", "horizontal host sweep", src, dst, r, max_p.get(fid, 0.0), {
                    "distinct_hosts": len(hs), "port": port, "window_s": WINDOW_S, "threshold": SWEEP_HOSTS,
                    "sample_hosts": sorted(hs)[:15]},
                    f"Scan: {src} probed port {port} on {len(hs)} different hosts within {int(WINDOW_S)} s "
                    f"(behavioural rule, threshold {SWEEP_HOSTS}).")))
        for (src, dst, port), sess in brute.items():
            if len(sess) >= BRUTE_CONN and self._cool(("brute", src, dst, port), now):
                fid, r = last[(src, dst)]
                out.append((fid, self._alert("BruteForce", f"{AUTH_PORTS[port]} brute force", src, dst, r,
                                             max_p.get(fid, 0.0), {
                    "connections": len(sess), "service": AUTH_PORTS[port], "port": port, "window_s": WINDOW_S,
                    "threshold": BRUTE_CONN},
                    f"BruteForce: {src} opened {len(sess)} short {AUTH_PORTS[port]} sessions to {dst}:{port} "
                    f"within {int(WINDOW_S)} s (behavioural rule, threshold {BRUTE_CONN}).")))
        return out

    @staticmethod
    def _alert(label, rule, src, dst, r: FlowRecord, p, evidence, summary) -> dict:
        return dict(severity="high", label=label, p_attack=float(p), ae_score=0.0, src_ip=src, dst_ip=dst,
                    dst_port=r.dst_port, proto=r.proto, summary=summary,
                    explanation={"detector": "behavioral", "rule": rule, "evidence": evidence, "contributions": []})
