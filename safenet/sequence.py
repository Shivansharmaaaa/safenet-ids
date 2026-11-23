"""Per-host sequence model: an LSTM over the last SEQ_LEN flows started by the same host.

A single flow of a port scan, a brute-force run or a slow DoS often looks harmless; the attack is in the
*sequence* (many short flows to new ports, the same login service hit again and again, regular gaps).
For every flow the model sees that flow plus the host's previous SEQ_LEN-1 flows, each described by the
22 shared flow features and three context features computed identically on datasets and live traffic:

    gap_s       whole seconds since the same host's previous flow started (UNSW-NB15's time resolution)
    same_dst    1 if the destination host equals the previous flow's
    same_dport  1 if the destination port equals the previous flow's

Windows are gathered by index at batch time, so millions of sequences never sit in memory at once.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import adversarial as adv
from .features import COMMON_FEATURES, FEATURE_DESCRIPTIONS, to_model_space

SEQ_LEN = 16
CONTEXT_FEATURES = ["gap_s", "same_dst", "same_dport"]
STEP_FEATURES = COMMON_FEATURES + CONTEXT_FEATURES + ["present"]
# Features that mostly measure network round-trip time: a LAN scan takes microseconds per probe, the same scan across
# the internet ~100s of ms. The timing-free variant cannot see them, so it must rely on structure and host context.
TIMING_FEATURES = ["duration_s", "bytes_per_s", "pkts_per_s", "fwd_iat_ms", "bwd_iat_ms"]
CONTEXT_DESCRIPTIONS = {
    "gap_s": "Gap since host's previous flow (s)",
    "same_dst": "Same destination host as previous flow",
    "same_dport": "Same destination port as previous flow",
    "present": "Flow present in window",
}


# ---------------------------------------------------------------------------------------- step features
def step_matrix(flows: pd.DataFrame, host: np.ndarray, ts: np.ndarray, dst: np.ndarray, dport: np.ndarray
                ) -> np.ndarray:
    """Step features for flows ALREADY sorted by (host, ts). Returns float32 [N, len(STEP_FEATURES)]."""
    n = len(flows)
    first = np.ones(n, dtype=bool)
    first[1:] = host[1:] != host[:-1]
    sec = np.floor(ts)                       # UNSW-NB15 records start times in whole seconds: use that everywhere
    gap = np.zeros(n, dtype=np.float64)
    gap[1:] = sec[1:] - sec[:-1]
    gap[first] = 0.0
    same_dst = np.zeros(n, dtype=np.float32)
    same_dst[1:] = dst[1:] == dst[:-1]
    same_dport = np.zeros(n, dtype=np.float32)
    same_dport[1:] = dport[1:] == dport[:-1]
    same_dst[first] = 0.0
    same_dport[first] = 0.0
    base = to_model_space(flows[COMMON_FEATURES].values)
    ctx = np.column_stack([np.log1p(np.clip(gap, 0, None)), same_dst, same_dport, np.ones(n)]).astype(np.float32)
    return np.hstack([base, ctx]).astype(np.float32)


def host_starts(host: np.ndarray) -> np.ndarray:
    """Index of the first row of each row's host block (rows sorted by host)."""
    n = len(host)
    first = np.ones(n, dtype=bool)
    first[1:] = host[1:] != host[:-1]
    starts = np.where(first, np.arange(n), 0)
    return np.maximum.accumulate(starts)


def gather(steps: np.ndarray, start: np.ndarray, idx: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Windows ending at rows idx: [B, seq_len, F], zero-padded (present=0) before the host's first flow."""
    offs = np.arange(seq_len - 1, -1, -1)
    rows = idx[:, None] - offs[None, :]
    valid = rows >= start[idx][:, None]
    win = steps[np.where(valid, rows, 0)]
    win[~valid] = 0.0
    return win


# ---------------------------------------------------------------------------------------- model
class SeqLSTM(nn.Module):
    def __init__(self, n_feat: int = len(STEP_FEATURES), hidden: int = 64, drop: list[str] | tuple = ()):
        super().__init__()
        mask = torch.ones(n_feat)
        for f in drop:
            mask[STEP_FEATURES.index(f)] = 0.0
        self.register_buffer("mask", mask)       # masked inside the model: identical in training, live and SHAP
        self.drop = list(drop)
        self.norm = nn.LayerNorm(n_feat)
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_feat, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x [B, T, F] -> logits [B]
        x = x * self.mask
        h, _ = self.lstm(self.norm(x))
        # the last hidden state summarises the context; the raw last step keeps the current flow sharp
        return self.head(torch.cat([h[:, -1], x[:, -1]], dim=1)).squeeze(1)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model: SeqLSTM, steps: np.ndarray, start: np.ndarray, idx: np.ndarray, batch: int = 8192,
            dev: torch.device | None = None) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(idx), batch):
        x = torch.from_numpy(gather(steps, start, idx[i:i + batch])).to(dev)
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train(steps: np.ndarray, start: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
          epochs: int = 4, batch: int = 4096, lr: float = 2e-3, seed: int = 42, log=print, drop=(),
          augment=None) -> SeqLSTM:
    """augment(windows, idx, rng, model) may rewrite a batch's windows in place (adversarial training)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = SeqLSTM(steps.shape[1], drop=drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = y[train_idx].mean()
    # sqrt-damped class weight, as for the tree models
    pos_w = torch.tensor(float(np.sqrt((1 - pos) / max(pos, 1e-6))), device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        tot = 0.0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            xb = gather(steps, start, b)
            if augment is not None:
                augment(xb, b, rng, model)
            x = torch.from_numpy(xb).to(dev)
            t = torch.from_numpy(y[b].astype(np.float32)).to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        p = predict(model, steps, start, val_idx)
        yv = y[val_idx].astype(np.float32)
        eps = 1e-6
        vloss = float(-np.mean(yv * np.log(p + eps) * float(pos_w) + (1 - yv) * np.log(1 - p + eps)))
        log(f"   epoch {ep + 1}/{epochs} train_loss={tot / len(order):.4f} val_loss={vloss:.4f}")
        if vloss < best:
            best, best_state = vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval()


# ---------------------------------------------------------------------------------------- adversarial training
ADV_FRAC, ADV_MAX_EPS = 0.5, 2.0


def perturb_windows(win: np.ndarray, raw: np.ndarray, y: np.ndarray, start: np.ndarray, idx: np.ndarray,
                    pad, slow, inj) -> np.ndarray:
    """The attacker's view of windows ending at idx: every ATTACK flow in each window is padded, slowed (longer
    flows and longer gaps before them) and given injected packets. Benign flows of the host are untouched.
    pad / slow / inj: scalars or one budget per window. Returns a new array."""
    K, F = SEQ_LEN, len(COMMON_FEATURES)
    offs = np.arange(K - 1, -1, -1)
    rows = idx[:, None] - offs[None, :]
    valid = rows >= start[idx][:, None]
    rows_c = np.where(valid, rows, 0)
    att = (y[rows_c] == 1) & valid
    per = lambda v: np.repeat(np.broadcast_to(np.asarray(v, dtype=np.float64), (len(idx),)), K)
    pert = adv.perturb(pd.DataFrame(raw[rows_c].reshape(-1, F), columns=COMMON_FEATURES),
                       per(pad), per(slow), per(inj)).values
    out = win.copy()
    out[..., :F] = np.where(att[..., None], to_model_space(pert).reshape(len(idx), K, F), out[..., :F])
    gap = np.expm1(out[..., F])
    slow_k = per(slow).reshape(len(idx), K)
    out[..., F] = np.where(att, np.log1p(np.floor(np.rint(gap) * (1 + slow_k) + 1e-6)), out[..., F])
    return out.astype(np.float32)


def adv_augmenter(raw: np.ndarray, y: np.ndarray, start: np.ndarray):
    """Adversarial training: half of each batch's attack windows are replaced by a random evasion attempt with a
    budget up to ADV_MAX_EPS (same perturbation family as the evaluation, random rather than adaptive)."""
    def aug(xb: np.ndarray, b: np.ndarray, rng: np.random.Generator, model=None) -> None:
        pos = np.where((y[b] == 1) & (rng.random(len(b)) < ADV_FRAC))[0]
        if len(pos) == 0:
            return
        eps = rng.uniform(0, ADV_MAX_EPS, len(pos))
        pad, slow, inj = rng.random((3, len(pos))) * eps
        xb[pos] = perturb_windows(xb[pos], raw, y, start, b[pos], pad, slow, inj)
    return aug


# the corners of the attacker's budget cube [0, eps]^3 (pad, slow, inject), without the unperturbed one
_CORNERS = np.array([(p, s, k) for p in (0, 1) for s in (0, 1) for k in (0, 1)][1:], dtype=np.float64)


def adaptive_augmenter(raw: np.ndarray, y: np.ndarray, start: np.ndarray):
    """Adaptive adversarial training: for half of each batch's attack windows, try the 7 corners of a random budget
    cube plus one random interior point against the CURRENT model and train on the attempt it scores lowest.
    The evaluation attacker searches a finer grid (64 points) at fixed budgets, so it stays the stronger one."""
    def aug(xb: np.ndarray, b: np.ndarray, rng: np.random.Generator, model=None) -> None:
        pos = np.where((y[b] == 1) & (rng.random(len(b)) < ADV_FRAC))[0]
        if len(pos) == 0 or model is None:
            return
        eps = rng.uniform(0, ADV_MAX_EPS, len(pos))
        cands = [_CORNERS[c][None, :] * eps[:, None] for c in range(len(_CORNERS))]
        cands.append(rng.random((len(pos), 3)) * eps[:, None])
        best_p, best_w = np.full(len(pos), np.inf), xb[pos].copy()
        dev = next(model.parameters()).device
        with torch.no_grad():
            for c in cands:
                w = perturb_windows(xb[pos], raw, y, start, b[pos], c[:, 0], c[:, 1], c[:, 2])
                p = torch.sigmoid(model(torch.from_numpy(w).to(dev))).cpu().numpy()
                better = p < best_p
                best_p[better], best_w[better] = p[better], w[better]
        xb[pos] = best_w
    return aug


# ---------------------------------------------------------------------------------------- explanation
def describe(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature) or CONTEXT_DESCRIPTIONS.get(feature, feature)


class SeqExplainer:
    """Expected-gradients SHAP (shap.GradientExplainer) for one window: per-feature and per-flow attribution."""

    def __init__(self, model: SeqLSTM, background: np.ndarray):
        import shap
        self.model = model.cpu().eval()
        self.bg = torch.from_numpy(background.astype(np.float32))
        self.exp = shap.GradientExplainer(_Logit(self.model), self.bg)

    def explain(self, window: np.ndarray, flows_meta: list[dict], top_k: int = 6) -> dict:
        x = torch.from_numpy(window[None].astype(np.float32))
        torch.backends.cudnn.enabled = False
        with torch.enable_grad():
            sv = self.exp.shap_values(x, nsamples=200)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(window.shape)   # [T, F]
        sv = sv.copy()
        sv[:, STEP_FEATURES.index("present")] = 0.0
        by_feat = sv.sum(axis=0)
        by_step = sv.sum(axis=1)
        last = window[-1]
        order = np.argsort(-np.abs(by_feat))[:top_k]
        contributions = []
        for j in order:
            name = STEP_FEATURES[j]
            raw = float(np.expm1(last[j])) if name in COMMON_FEATURES or name == "gap_s" else float(last[j])
            contributions.append({"feature": name, "description": describe(name), "value": raw,
                                  "shap": float(by_feat[j])})
        with torch.no_grad():
            base = float(_Logit(self.model)(self.bg).mean())
        steps = []
        n_present = int(window[:, STEP_FEATURES.index("present")].sum())
        for k, meta in enumerate(flows_meta[-n_present:] if n_present else []):
            t = len(window) - n_present + k
            steps.append({**meta, "shap": float(by_step[t])})
        return {"detector": "sequence", "base_value": base, "contributions": contributions, "steps": steps,
                "seq_len": len(window), "context_flows": n_present}


class _Logit(nn.Module):
    def __init__(self, m: SeqLSTM):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).unsqueeze(1)


# ---------------------------------------------------------------------------------------- live state
@dataclass
class _Step:
    vec: np.ndarray
    ts: float
    dst: str
    dport: int
    meta: dict


class HostHistory:
    """Live per-host flow history, producing windows identical to training."""

    def __init__(self, seq_len: int = SEQ_LEN, idle_s: float = 3600.0):
        self.seq_len, self.idle_s = seq_len, idle_s
        self.h: dict[str, deque[_Step]] = {}

    def push(self, host: str, ts: float, dst: str, dport: int, flow_feats: np.ndarray, meta: dict
             ) -> tuple[np.ndarray, list[dict]]:
        q = self.h.setdefault(host, deque(maxlen=self.seq_len))
        prev = q[-1] if q else None
        gap = max(0.0, np.floor(ts) - np.floor(prev.ts)) if prev else 0.0
        ctx = np.array([np.log1p(gap), float(prev is not None and prev.dst == dst),
                        float(prev is not None and prev.dport == dport), 1.0], dtype=np.float32)
        vec = np.concatenate([to_model_space(flow_feats[None])[0], ctx]).astype(np.float32)
        q.append(_Step(vec, ts, dst, dport, meta))
        win = np.zeros((self.seq_len, len(STEP_FEATURES)), dtype=np.float32)
        for k, s in enumerate(q):
            win[self.seq_len - len(q) + k] = s.vec
        return win, [s.meta for s in q]

    def prune(self, now: float) -> None:
        for host in [h for h, q in self.h.items() if q and now - q[-1].ts > self.idle_s]:
            del self.h[host]
