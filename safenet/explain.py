"""SHAP explanations: global importance for reports and per-alert local explanations."""
from __future__ import annotations

import numpy as np
import shap

from .features import FEATURE_DESCRIPTIONS


def _positive_class_values(sv) -> np.ndarray:
    """Normalise SHAP output to (n, features) for the attack / positive class."""
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, -1]
    return sv


def global_importance(model, X_model: np.ndarray, feature_names: list[str], n: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_model), size=min(n, len(X_model)), replace=False)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_model[idx])
    if isinstance(raw, list):                       # old shap multiclass: [classes] x (n, f)
        imp = np.mean([np.abs(a).mean(axis=0) for a in raw], axis=0)
    else:
        arr = np.abs(np.asarray(raw))
        imp = arr.mean(axis=(0, 2)) if arr.ndim == 3 else arr.mean(axis=0)   # (n, f, c) or (n, f)
    order = np.argsort(-imp)
    return [(feature_names[i], float(imp[i])) for i in order]


class AlertExplainer:
    """Explains why the binary detector called a flow malicious."""

    def __init__(self, binary_model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(binary_model)
        self.names = feature_names
        ev = self.explainer.expected_value
        self.base = float(np.ravel(ev)[-1])

    def explain(self, x_model_row: np.ndarray, raw_row: np.ndarray, top_k: int = 6) -> dict:
        sv = _positive_class_values(self.explainer.shap_values(x_model_row.reshape(1, -1)))[0]
        order = np.argsort(-np.abs(sv))[:top_k]
        contribs = [
            {
                "feature": self.names[i],
                "description": FEATURE_DESCRIPTIONS.get(self.names[i], self.names[i]),
                "value": float(raw_row[i]),
                "shap": float(sv[i]),
            }
            for i in order
        ]
        return {"base_value": self.base, "contributions": contribs}

    @staticmethod
    def summarise(label: str, expl: dict) -> str:
        pushes = [c for c in expl["contributions"] if c["shap"] > 0][:3]
        if not pushes:
            return f"{label}: flagged by anomaly detector (unusual for benign traffic)."
        parts = [f"{c['description'].lower()} = {_fmt(c['value'])}" for c in pushes]
        return f"{label}: driven mainly by " + "; ".join(parts) + "."


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.3g}"
