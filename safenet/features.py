"""SafeNet common flow feature schema.

CIC-IDS2017 (CICFlowMeter, 78 features) and UNSW-NB15 (Argus/Bro, 49 features)
describe flows with different tools and different columns. To (a) test a model
trained on one dataset against the other and (b) score live traffic captured on
this machine, SafeNet maps every flow into ONE shared set of features that can
be computed identically from:

  * CIC-IDS2017 CSV columns          -> ``from_cicids2017``
  * UNSW-NB15 CSV columns            -> ``from_unswnb15``
  * live packets (safenet.flowmeter) -> ``FlowRecord.features``

Direction convention: "fwd" = the side that sent the first packet (initiator),
"bwd" = the responder. Byte counts are IP-level (headers + payload).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_FEATURES: list[str] = [
    "duration_s",       # flow duration in seconds
    "fwd_pkts",         # packets initiator -> responder
    "bwd_pkts",         # packets responder -> initiator
    "fwd_bytes",        # IP bytes initiator -> responder
    "bwd_bytes",        # IP bytes responder -> initiator
    "fwd_pkt_mean",     # mean IP packet size, fwd
    "bwd_pkt_mean",     # mean IP packet size, bwd
    "bytes_per_s",      # (fwd_bytes + bwd_bytes) / duration
    "pkts_per_s",       # (fwd_pkts + bwd_pkts) / duration
    "fwd_iat_ms",       # mean inter-arrival time of fwd packets (ms)
    "bwd_iat_ms",       # mean inter-arrival time of bwd packets (ms)
    "fwd_win",          # first TCP window advertised by initiator (0 if n/a)
    "bwd_win",          # first TCP window advertised by responder (0 if n/a)
    "is_tcp",           # 1 if TCP flow
    "byte_ratio",       # bwd_bytes / (fwd_bytes + 1)
    "svc_http", "svc_tls", "svc_dns", "svc_ssh", "svc_ftp", "svc_mail", "svc_other",
]

FEATURE_DESCRIPTIONS = {
    "duration_s": "Flow duration (s)",
    "fwd_pkts": "Packets sent by initiator",
    "bwd_pkts": "Packets sent by responder",
    "fwd_bytes": "Bytes sent by initiator",
    "bwd_bytes": "Bytes sent by responder",
    "fwd_pkt_mean": "Avg packet size from initiator",
    "bwd_pkt_mean": "Avg packet size from responder",
    "bytes_per_s": "Throughput (bytes/s)",
    "pkts_per_s": "Packet rate (pkts/s)",
    "fwd_iat_ms": "Gap between initiator packets (ms)",
    "bwd_iat_ms": "Gap between responder packets (ms)",
    "fwd_win": "Initiator TCP window",
    "bwd_win": "Responder TCP window",
    "is_tcp": "TCP flow",
    "byte_ratio": "Response/request byte ratio",
    "svc_http": "HTTP service",
    "svc_tls": "TLS/HTTPS service",
    "svc_dns": "DNS service",
    "svc_ssh": "SSH service",
    "svc_ftp": "FTP service",
    "svc_mail": "Mail service",
    "svc_other": "Other service",
}

# Harmonised attack categories shared by both datasets and the live sensor.
CATEGORIES = ["Benign", "DoS", "Scan", "BruteForce", "WebAttack", "Botnet", "Exploit"]

CIC_LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS", "DDoS": "DoS",
    "PortScan": "Scan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack Brute Force": "WebAttack", "Web Attack XSS": "WebAttack",
    "Web Attack Sql Injection": "WebAttack",
    "Bot": "Botnet",
    "Infiltration": "Exploit", "Heartbleed": "Exploit",
}

UNSW_LABEL_MAP = {
    "Normal": "Benign",
    "DoS": "DoS",
    "Reconnaissance": "Scan", "Analysis": "Scan",
    "Fuzzers": "Exploit",
    "Exploits": "Exploit", "Backdoor": "Exploit", "Shellcode": "Exploit",
    "Worms": "Exploit", "Generic": "Exploit",
}

_SVC_PORTS = {
    "svc_http": {80, 8080, 8000, 8008, 8888},
    "svc_tls": {443, 8443, 993, 995, 465},
    "svc_dns": {53, 5353},
    "svc_ssh": {22},
    "svc_ftp": {20, 21},
    "svc_mail": {25, 110, 143, 587},
}
_UNSW_SVC = {
    "http": "svc_http", "ssl": "svc_tls", "dns": "svc_dns", "ssh": "svc_ssh",
    "ftp": "svc_ftp", "ftp-data": "svc_ftp", "smtp": "svc_mail", "pop3": "svc_mail",
}
SVC_COLS = [c for c in COMMON_FEATURES if c.startswith("svc_")]


def service_from_port(port: int) -> str:
    for col, ports in _SVC_PORTS.items():
        if port in ports:
            return col
    return "svc_other"


def _svc_onehot(svc: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({c: (svc == c).astype(np.float32) for c in SVC_COLS}, index=svc.index)


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    dur = df["duration_s"].clip(lower=1e-6)
    total_b = df["fwd_bytes"] + df["bwd_bytes"]
    total_p = df["fwd_pkts"] + df["bwd_pkts"]
    df["bytes_per_s"] = total_b / dur
    df["pkts_per_s"] = total_p / dur
    df["byte_ratio"] = df["bwd_bytes"] / (df["fwd_bytes"] + 1.0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    num = [c for c in COMMON_FEATURES if c not in SVC_COLS and c != "is_tcp"]
    df[num] = df[num].clip(lower=0)
    return df[COMMON_FEATURES].astype(np.float32)


def from_cicids2017(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw.rename(columns=lambda c: c.strip())
    fwd_hdr = r["Fwd Header Length"].clip(lower=0)
    bwd_hdr = r["Bwd Header Length"].clip(lower=0)
    out = pd.DataFrame(index=r.index)
    out["duration_s"] = r["Flow Duration"].clip(lower=0) / 1e6
    out["fwd_pkts"] = r["Total Fwd Packets"]
    out["bwd_pkts"] = r["Total Backward Packets"]
    out["fwd_bytes"] = r["Total Length of Fwd Packets"] + fwd_hdr
    out["bwd_bytes"] = r["Total Length of Bwd Packets"] + bwd_hdr
    out["fwd_pkt_mean"] = out["fwd_bytes"] / out["fwd_pkts"].clip(lower=1)
    out["bwd_pkt_mean"] = out["bwd_bytes"] / out["bwd_pkts"].clip(lower=1)
    out["fwd_iat_ms"] = r["Fwd IAT Mean"].clip(lower=0) / 1e3
    out["bwd_iat_ms"] = r["Bwd IAT Mean"].clip(lower=0) / 1e3
    out["fwd_win"] = r["Init_Win_bytes_forward"].clip(lower=0)
    out["bwd_win"] = r["Init_Win_bytes_backward"].clip(lower=0)
    flags = r[["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"]].sum(axis=1)
    out["is_tcp"] = ((r["Init_Win_bytes_forward"] >= 0) | (r["Init_Win_bytes_backward"] >= 0) | (flags > 0)).astype(np.float32)
    svc = r["Destination Port"].astype(int).map(service_from_port)
    out = pd.concat([out, _svc_onehot(svc)], axis=1)
    return _finish(out)


def from_unswnb15(raw: pd.DataFrame) -> pd.DataFrame:
    r = raw
    out = pd.DataFrame(index=r.index)
    out["duration_s"] = r["dur"]
    out["fwd_pkts"] = r["spkts"]
    out["bwd_pkts"] = r["dpkts"]
    out["fwd_bytes"] = r["sbytes"]
    out["bwd_bytes"] = r["dbytes"]
    out["fwd_pkt_mean"] = r["sbytes"] / r["spkts"].clip(lower=1)
    out["bwd_pkt_mean"] = r["dbytes"] / r["dpkts"].clip(lower=1)
    out["fwd_iat_ms"] = r["sinpkt"]
    out["bwd_iat_ms"] = r["dinpkt"]
    out["fwd_win"] = r["swin"]
    out["bwd_win"] = r["dwin"]
    out["is_tcp"] = (r["proto"].astype(str).str.lower() == "tcp").astype(np.float32)
    svc = r["service"].astype(str).map(_UNSW_SVC).fillna("svc_other")
    out = pd.concat([out, _svc_onehot(svc)], axis=1)
    return _finish(out)


def to_model_space(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """log1p-compress heavy-tailed counts so tree + neural models see similar scales."""
    arr = np.asarray(X, dtype=np.float32)
    return np.log1p(np.clip(arr, 0, None))
