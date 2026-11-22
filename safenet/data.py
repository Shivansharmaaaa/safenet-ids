"""Loading + cleaning of the real CIC-IDS2017 and UNSW-NB15 releases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .features import CIC_LABEL_MAP, UNSW_LABEL_MAP, from_cicids2017, from_unswnb15

RAW = DATA_DIR / "raw"
PROC = DATA_DIR / "processed"


def _cic_files() -> list[Path]:
    files = sorted((RAW / "cicids2017").glob("*.pcap_ISCX.csv"))
    if len(files) != 8:
        raise FileNotFoundError(f"expected 8 CIC-IDS2017 MachineLearningCSV files, found {len(files)}")
    return files


def load_cicids2017_native() -> pd.DataFrame:
    """All 8 days, cleaned. Returns native 78 features + 'label' + 'category'."""
    cache = PROC / "cicids2017_native.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for f in _cic_files():
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["day"] = f.name.split(".")[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]  # "Fwd Header Length" appears twice
    # the original files contain a mis-encoded dash in "Web Attack – XSS" etc.
    df["Label"] = df["Label"].astype(str).str.strip().str.replace(
        r"^Web Attack.*?(Brute Force|XSS|Sql Injection)$", r"Web Attack \1", regex=True)
    feat_cols = [c for c in df.columns if c not in ("Label", "day")]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    df["category"] = df["Label"].map(CIC_LABEL_MAP)
    unknown = df["category"].isna()
    if unknown.any():
        raise ValueError(f"unmapped CIC labels: {df.loc[unknown, 'Label'].unique()}")
    df = df.rename(columns={"Label": "label"})
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def load_unswnb15_native() -> pd.DataFrame:
    """Official UNSW-NB15 training + testing partitions with a 'split' column."""
    cache = PROC / "unswnb15_native.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # The official release has 175,341 training and 82,332 testing records.
    # Some mirrors swap the file names, so assign the split by size.
    loaded = [pd.read_csv(RAW / "unswnb15" / f"{n}.csv", low_memory=False, encoding="utf-8-sig")
              for n in ("train", "test")]
    loaded.sort(key=len, reverse=True)
    if [len(d) for d in loaded] != [175341, 82332]:
        raise ValueError(f"unexpected UNSW-NB15 sizes {[len(d) for d in loaded]}")
    parts = []
    for name, df in zip(("train", "test"), loaded):
        df["split"] = name
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df.loc[df["attack_cat"].isin(["", "-", "nan"]), "attack_cat"] = "Normal"
    df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})
    df["category"] = df["attack_cat"].map(UNSW_LABEL_MAP)
    unknown = df["category"].isna()
    if unknown.any():
        raise ValueError(f"unmapped UNSW labels: {df.loc[unknown, 'attack_cat'].unique()}")
    df = df.drop(columns=["label"]).rename(columns={"attack_cat": "label"})  # binary label is implied by category
    PROC.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


CIC_IMPROVED_MAP = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS Slowloris": "DoS", "DoS Slowhttptest": "DoS", "DDoS": "DoS",
    "Portscan": "Scan", "Infiltration - Portscan": "Scan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack", "Web Attack - XSS": "WebAttack", "Web Attack - SQL Injection": "WebAttack",
    "Botnet": "Botnet",
    "Infiltration": "Exploit", "Heartbleed": "Exploit",
}
_CIC_IMPROVED_COLS = {
    "Dst Port": "Destination Port", "Total Fwd Packet": "Total Fwd Packets", "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets", "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "FWD Init Win Bytes": "Init_Win_bytes_forward", "Bwd Init Win Bytes": "Init_Win_bytes_backward",
}


def load_cicids2017_full() -> pd.DataFrame:
    """CIC-IDS2017 as re-extracted and re-labelled by Engelen et al. (KU Leuven DistriNet, 2021/2022): 2,099,976 flows
    with source/destination IPs and microsecond start times, needed for per-host sequences. Flows labelled
    '... - Attempted' (the attack tool ran but the flow shows no malicious behaviour) count as Benign, as the
    dataset authors recommend. Returns the shared-feature inputs plus host, ts, dst, dport, label, category."""
    cache = PROC / "cicids2017_full.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    need = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packet",
            "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet", "Fwd Header Length",
            "Bwd Header Length", "Fwd IAT Mean", "Bwd IAT Mean", "FWD Init Win Bytes", "Bwd Init Win Bytes",
            "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "Label"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    df = pd.concat([pd.read_csv(RAW / "cicids2017_improved" / f"{d}.csv", usecols=need, low_memory=False)
                    for d in days], ignore_index=True)
    if len(df) != 2_099_976:
        raise ValueError(f"expected the 2,099,976 flows of CICIDS2017_improved, got {len(df):,}")
    lab = df["Label"].astype(str).str.strip()
    lab = lab.where(~lab.str.endswith(" - Attempted"), "BENIGN")
    df["label"] = lab
    df["category"] = lab.map(CIC_IMPROVED_MAP)
    unknown = df["category"].isna()
    if unknown.any():
        raise ValueError(f"unmapped CIC labels: {df.loc[unknown, 'label'].unique()}")
    df["proto"] = df["Protocol"].map({6: "tcp", 17: "udp", 1: "icmp"}).fillna("other")
    df["host"] = df["Src IP"].astype(str)
    df["dst"] = df["Dst IP"].astype(str)
    df["dport"] = pd.to_numeric(df["Dst Port"], errors="coerce").fillna(0).astype(np.int64)
    df["ts"] = (pd.to_datetime(df["Timestamp"]) - pd.Timestamp("1970-01-01")).dt.total_seconds()  # unit-safe
    df = df.drop(columns=["Label", "Src IP", "Dst IP", "Timestamp", "Protocol"]).rename(columns=_CIC_IMPROVED_COLS)
    num = [c for c in df.columns if c not in ("label", "category", "proto", "host", "dst", "dport", "ts")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df.to_parquet(cache, index=False)
    return df


def _port(s: pd.Series) -> pd.Series:
    """Ports as int; the UNSW CSVs contain some hex ('0x000b') and '-' entries."""
    s = s.astype(str).str.strip()
    hexa = s.str.startswith("0x")
    out = pd.to_numeric(s.where(~hexa), errors="coerce")
    out[hexa] = s[hexa].map(lambda v: int(v, 16))
    return out.fillna(0).astype(np.int64)


def load_unswnb15_full() -> pd.DataFrame:
    """The full official UNSW-NB15 records (UNSW-NB15_1..4.csv, 2,540,047 flows) WITH source/destination IPs and
    start times, needed to build per-host flow sequences. Returns the shared-feature inputs plus
    host, ts, dst, dport, label, category."""
    cache = PROC / "unswnb15_full.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    d = RAW / "unswnb15_full"
    names = pd.read_csv(d / "NUSW-NB15_features.csv", encoding="latin-1")["Name"].str.replace(" ", "").tolist()
    keep = ["srcip", "sport", "dstip", "dsport", "proto", "dur", "sbytes", "dbytes", "service", "Spkts", "Dpkts",
            "swin", "dwin", "Stime", "Sintpkt", "Dintpkt", "attack_cat"]
    parts = [pd.read_csv(d / f"UNSW-NB15_{i}.csv", header=None, names=names, usecols=keep, low_memory=False,
                         encoding="latin-1") for i in range(1, 5)]
    df = pd.concat(parts, ignore_index=True)
    if len(df) != 2_540_047:
        raise ValueError(f"expected the 2,540,047 official UNSW-NB15 records, got {len(df):,}")
    df = df.rename(columns={"Spkts": "spkts", "Dpkts": "dpkts", "Sintpkt": "sinpkt", "Dintpkt": "dinpkt"})
    lab = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    lab = lab.replace({"": "Normal", "-": "Normal", "nan": "Normal", "Backdoors": "Backdoor"})
    df["label"] = lab
    df["category"] = lab.map(UNSW_LABEL_MAP)
    unknown = df["category"].isna()
    if unknown.any():
        raise ValueError(f"unmapped UNSW labels: {df.loc[unknown, 'label'].unique()}")
    df["host"] = df["srcip"].astype(str)
    df["dst"] = df["dstip"].astype(str)
    df["dport"] = _port(df["dsport"])
    df["ts"] = df["Stime"].astype(np.float64)
    df = df.drop(columns=["attack_cat", "srcip", "dstip", "dsport", "sport", "Stime"])
    df.to_parquet(cache, index=False)
    return df


def common_view(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X_common, category, original_label) for a dataset name."""
    if name == "cicids2017":
        df = load_cicids2017_native()
        return from_cicids2017(df), df["category"], df["label"]
    if name == "unswnb15":
        df = load_unswnb15_native()
        return from_unswnb15(df), df["category"], df["label"]
    raise ValueError(name)
