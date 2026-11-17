"""Adversarial (evasion) robustness testing in a realistic threat model.

The attacker controls only the packets THEY send (the fwd direction) and can:
  * pad      - pad payloads (packet size grows, capped at a 1500 B MTU)
  * slow     - slow the attack down (longer duration / inter-arrival gaps)
  * inject   - inject extra small dummy packets (60 B each)
They cannot change what the victim (bwd direction) sends. Every perturbation is
applied in raw feature space and all derived features are recomputed, so the
perturbed flow is one that could really exist on the wire.

``adaptive_attack`` is a black-box, model-guided attack: for every malicious
flow it searches a grid of (pad, slow, inject) combinations within budget
``eps`` and keeps the one the target model finds least suspicious.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from .features import COMMON_FEATURES

MTU = 1500.0


def perturb(X: pd.DataFrame, pad: float = 0.0, slow: float = 0.0, inject: float = 0.0) -> pd.DataFrame:
    P = X.copy()
    if pad:
        new_mean = np.minimum(P["fwd_pkt_mean"] * (1 + pad), MTU)
        new_mean = np.maximum(new_mean, P["fwd_pkt_mean"])
        P["fwd_pkt_mean"] = new_mean
        P["fwd_bytes"] = new_mean * P["fwd_pkts"]
    if inject:
        extra = np.ceil(P["fwd_pkts"] * inject)
        P["fwd_bytes"] = P["fwd_bytes"] + 60.0 * extra
        P["fwd_pkts"] = P["fwd_pkts"] + extra
        P["fwd_pkt_mean"] = P["fwd_bytes"] / P["fwd_pkts"].clip(lower=1)
        P["fwd_iat_ms"] = np.where(P["fwd_pkts"] > 1, P["duration_s"] * 1000.0 / (P["fwd_pkts"] - 1), P["fwd_iat_ms"])
    if slow:
        P["duration_s"] = P["duration_s"] * (1 + slow)
        P["fwd_iat_ms"] = P["fwd_iat_ms"] * (1 + slow)
        P["bwd_iat_ms"] = P["bwd_iat_ms"] * (1 + slow)
    dur = P["duration_s"].clip(lower=1e-6)
    P["bytes_per_s"] = (P["fwd_bytes"] + P["bwd_bytes"]) / dur
    P["pkts_per_s"] = (P["fwd_pkts"] + P["bwd_pkts"]) / dur
    P["byte_ratio"] = P["bwd_bytes"] / (P["fwd_bytes"] + 1.0)
    return P[COMMON_FEATURES].astype(np.float32)


def grid(eps: float, steps: int = 4) -> list[tuple[float, float, float]]:
    vals = np.linspace(0, eps, steps)
    return [c for c in itertools.product(vals, vals, vals)]


def adaptive_attack(X_attack: pd.DataFrame, attack_prob: Callable[[pd.DataFrame], np.ndarray], eps: float,
                    steps: int = 4) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the most evasive perturbation per flow and its attack probability."""
    best_p = attack_prob(X_attack)
    best_X = X_attack.copy()
    if eps <= 0:
        return best_X, best_p
    for pad, slow, inj in grid(eps, steps):
        if pad == slow == inj == 0:
            continue
        Xp = perturb(X_attack, pad, slow, inj)
        p = attack_prob(Xp)
        better = p < best_p
        if better.any():
            best_p = np.where(better, p, best_p)
            best_X.loc[better] = Xp.loc[better].values
    return best_X, best_p
