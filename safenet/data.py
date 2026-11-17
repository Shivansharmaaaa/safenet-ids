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


def common_view(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X_common, category, original_label) for a dataset name."""
    if name == "cicids2017":
        df = load_cicids2017_native()
        return from_cicids2017(df), df["category"], df["label"]
    if name == "unswnb15":
        df = load_unswnb15_native()
        return from_unswnb15(df), df["category"], df["label"]
    raise ValueError(name)
