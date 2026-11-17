"""Live bidirectional flow meter.

Aggregates captured packets into flows keyed by the 5-tuple and produces the
exact COMMON_FEATURES used to train the models. Flow termination mirrors
CICFlowMeter: TCP FIN/RST, an idle timeout, or an active timeout.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .features import COMMON_FEATURES, SVC_COLS, service_from_port

IDLE_TIMEOUT = 15.0
ACTIVE_TIMEOUT = 120.0
FIN_GRACE = 1.0


@dataclass
class FlowRecord:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    first_ts: float
    last_ts: float
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_last: float | None = None
    bwd_last: float | None = None
    fwd_iat_sum: float = 0.0
    bwd_iat_sum: float = 0.0
    fwd_win: int = -1
    bwd_win: int = -1
    closed_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, ts: float, fwd: bool, size: int, win: int | None, fin_rst: bool) -> None:
        self.last_ts = max(self.last_ts, ts)
        if fwd:
            if self.fwd_last is not None:
                self.fwd_iat_sum += max(0.0, ts - self.fwd_last)
            self.fwd_last = ts
            self.fwd_pkts += 1
            self.fwd_bytes += size
            if win is not None and self.fwd_win < 0:
                self.fwd_win = win
        else:
            if self.bwd_last is not None:
                self.bwd_iat_sum += max(0.0, ts - self.bwd_last)
            self.bwd_last = ts
            self.bwd_pkts += 1
            self.bwd_bytes += size
            if win is not None and self.bwd_win < 0:
                self.bwd_win = win
        if fin_rst and self.closed_at is None:
            self.closed_at = ts

    def features(self) -> dict[str, float]:
        dur = max(self.last_ts - self.first_ts, 0.0)
        f = {
            "duration_s": dur,
            "fwd_pkts": float(self.fwd_pkts),
            "bwd_pkts": float(self.bwd_pkts),
            "fwd_bytes": float(self.fwd_bytes),
            "bwd_bytes": float(self.bwd_bytes),
            "fwd_pkt_mean": self.fwd_bytes / max(self.fwd_pkts, 1),
            "bwd_pkt_mean": self.bwd_bytes / max(self.bwd_pkts, 1),
            "fwd_iat_ms": 1000.0 * self.fwd_iat_sum / max(self.fwd_pkts - 1, 1) if self.fwd_pkts > 1 else 0.0,
            "bwd_iat_ms": 1000.0 * self.bwd_iat_sum / max(self.bwd_pkts - 1, 1) if self.bwd_pkts > 1 else 0.0,
            "fwd_win": float(max(self.fwd_win, 0)),
            "bwd_win": float(max(self.bwd_win, 0)),
            "is_tcp": 1.0 if self.proto == "tcp" else 0.0,
        }
        d = max(dur, 1e-6)
        f["bytes_per_s"] = (self.fwd_bytes + self.bwd_bytes) / d
        f["pkts_per_s"] = (self.fwd_pkts + self.bwd_pkts) / d
        f["byte_ratio"] = self.bwd_bytes / (self.fwd_bytes + 1.0)
        svc = service_from_port(self.dst_port)
        for c in SVC_COLS:
            f[c] = 1.0 if c == svc else 0.0
        return {k: float(f[k]) for k in COMMON_FEATURES}

    def vector(self) -> np.ndarray:
        feats = self.features()
        return np.array([feats[k] for k in COMMON_FEATURES], dtype=np.float32)


class FlowTable:
    def __init__(self, ignore_ips: set[str] | None = None):
        self.flows: dict[tuple, FlowRecord] = {}
        self.lock = threading.Lock()
        self.packets = 0
        self.ignore = ignore_ips or set()

    def on_packet(self, pkt) -> None:
        from scapy.layers.inet import ICMP, IP, TCP, UDP
        from scapy.layers.inet6 import IPv6

        if IP in pkt:
            ip = pkt[IP]
            src, dst, size = ip.src, ip.dst, int(ip.len)
        elif IPv6 in pkt:
            ip = pkt[IPv6]
            src, dst, size = ip.src, ip.dst, int(ip.plen) + 40
        else:
            return
        if src in self.ignore or dst in self.ignore:
            return
        win, fin_rst = None, False
        if TCP in pkt:
            l4, proto = pkt[TCP], "tcp"
            sport, dport = int(l4.sport), int(l4.dport)
            win = int(l4.window)
            flags = int(l4.flags)
            fin_rst = bool(flags & 0x01 or flags & 0x04)
        elif UDP in pkt:
            l4, proto = pkt[UDP], "udp"
            sport, dport = int(l4.sport), int(l4.dport)
        elif ICMP in pkt:
            proto, sport, dport = "icmp", 0, 0
        else:
            proto, sport, dport = f"ip{getattr(ip, 'proto', getattr(ip, 'nh', 0))}", 0, 0
        ts = float(pkt.time)
        a, b = (src, sport), (dst, dport)
        key = (proto,) + ((a, b) if a <= b else (b, a))
        with self.lock:
            self.packets += 1
            rec = self.flows.get(key)
            if rec is None:
                rec = FlowRecord(src, sport, dst, dport, proto, ts, ts)
                self.flows[key] = rec
            fwd = (src == rec.src_ip and sport == rec.src_port)
            rec.add(ts, fwd, size, win, fin_rst)

    def expire(self, now: float, flush_all: bool = False) -> list[FlowRecord]:
        done = []
        with self.lock:
            for key, rec in list(self.flows.items()):
                if (flush_all
                        or (rec.closed_at is not None and now - rec.closed_at >= FIN_GRACE)
                        or now - rec.last_ts >= IDLE_TIMEOUT
                        or now - rec.first_ts >= ACTIVE_TIMEOUT):
                    done.append(self.flows.pop(key))
        return done
