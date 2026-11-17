"""Model zoo: Random Forest, XGBoost (supervised) and a PyTorch Autoencoder
(unsupervised anomaly detector trained on benign traffic only, which lets it
flag rare / unseen attacks that the supervised models never saw)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .config import SEED


def make_rf(n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=150, max_depth=24, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=n_jobs, random_state=SEED,
    )


def make_xgb(n_classes: int) -> XGBClassifier:
    params = dict(
        n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, tree_method="hist", n_jobs=-1, random_state=SEED,
        max_bin=256,
    )
    if n_classes > 2:
        return XGBClassifier(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss", **params)
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Class-balanced sample weights (sqrt-damped) so rare attacks are not ignored."""
    w = compute_sample_weight("balanced", y)
    return np.sqrt(w)


class _AE(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        h = max(8, d // 2)
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, h), torch.nn.ReLU(), torch.nn.Linear(h, 6),
        )
        self.dec = torch.nn.Sequential(
            torch.nn.Linear(6, h), torch.nn.ReLU(), torch.nn.Linear(h, 64), torch.nn.ReLU(), torch.nn.Linear(64, d),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderDetector:
    """Train on benign flows only; anomaly score = reconstruction MSE.
    Threshold = the given quantile of benign validation error."""

    def __init__(self, epochs: int = 15, batch: int = 2048, lr: float = 1e-3, quantile: float = 0.99):
        self.epochs, self.batch, self.lr, self.quantile = epochs, batch, lr, quantile
        self.scaler = StandardScaler()
        self.model: _AE | None = None
        self.threshold: float = float("nan")

    def fit(self, X_benign: np.ndarray, X_val_benign: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(SEED)
        Xs = self.scaler.fit_transform(X_benign).astype(np.float32)
        self.model = _AE(Xs.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        t = torch.from_numpy(Xs)
        g = torch.Generator().manual_seed(SEED)
        for _ in range(self.epochs):
            perm = torch.randperm(len(t), generator=g)
            for i in range(0, len(t), self.batch):
                xb = t[perm[i:i + self.batch]]
                loss = torch.mean((self.model(xb) - xb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.threshold = float(np.quantile(self.score(X_val_benign), self.quantile))
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        out = []
        Xs = self.scaler.transform(X).astype(np.float32)
        for i in range(0, len(Xs), 65536):
            xb = torch.from_numpy(Xs[i:i + 65536])
            out.append(torch.mean((self.model(xb) - xb) ** 2, dim=1).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score(X) > self.threshold).astype(int)

    # torch modules pickle fine via joblib; keep a plain-state path for portability
    def __getstate__(self):
        s = self.__dict__.copy()
        s["model_state"] = None if self.model is None else {k: v.cpu() for k, v in self.model.state_dict().items()}
        s["d"] = None if self.model is None else self.model.enc[0].in_features
        s.pop("model")
        return s

    def __setstate__(self, s):
        state, d = s.pop("model_state"), s.pop("d")
        self.__dict__.update(s)
        self.model = None
        if state is not None:
            self.model = _AE(d)
            self.model.load_state_dict(state)
