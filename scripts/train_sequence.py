"""Per-host LSTM sequence model: training, evaluation against per-flow XGBoost, cross-dataset tests,
adaptive evasion test, and the live sequence bundle.

Data: the two releases that keep source IPs and start times, so each flow can be placed in its host's history:
  * CIC-IDS2017, re-extracted and re-labelled by Engelen et al. (KU Leuven DistriNet): 2,099,976 flows
  * UNSW-NB15 full records (UNSW-NB15_1..4.csv, UNSW Canberra): 2,540,047 flows

Split: global 10-minute time buckets assigned at random (70 % train / 10 % validation / 20 % test, fixed seed).
The per-flow XGBoost baseline is trained and tested on exactly the same rows, so the comparison isolates what the
host context adds. Only TCP/UDP flows enter histories, as on the live sensor.

usage: .venv/bin/python -m scripts.train_sequence [--epochs 4] [--no-db]
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from safenet import adversarial as adv
from safenet import db
from safenet import sequence as sq
from safenet.config import MODELS_DIR, REPORTS_DIR, SEED, database_url
from safenet.data import load_cicids2017_full, load_unswnb15_full
from safenet.features import COMMON_FEATURES, from_cicids2017, from_unswnb15, to_model_space
from safenet.models import balanced_weights, make_xgb

T0 = time.time()
RNG = np.random.default_rng(SEED)
RESULTS: list[dict] = []
NAMES = {"cic": "cicids2017", "unsw": "unswnb15", "both": "combined"}
# "LSTM" sees all 22 flow features; "LSTM-NoTiming" cannot see the five that mostly measure network round-trip time
VARIANTS = {"LSTM": [], "LSTM-NoTiming": sq.TIMING_FEATURES, "LSTM-NoTiming-Adv": sq.TIMING_FEATURES,
            "LSTM-NoTiming-AdvAdaptive": sq.TIMING_FEATURES}
# trained on adversarially perturbed attack windows: a random attempt, or the worst of several against the model itself
ADVERSARIAL = {"LSTM-NoTiming-Adv": "random", "LSTM-NoTiming-AdvAdaptive": "adaptive"}
ADV_FRAC, ADV_MAX_EPS = 0.5, 2.0


def perturb_windows(win: np.ndarray, raw: np.ndarray, y: np.ndarray, start: np.ndarray, idx: np.ndarray,
                    pad, slow, inj) -> np.ndarray:
    """The attacker's view of windows ending at idx: every ATTACK flow in each window is padded, slowed (longer
    flows and longer gaps before them) and given injected packets. Benign flows of the host are untouched.
    pad / slow / inj: scalars or one budget per window. Returns a new array."""
    K, F = sq.SEQ_LEN, len(COMMON_FEATURES)
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    m = dict(precision=prec, recall=rec, f1=2 * prec * rec / max(1e-12, prec + rec), fpr=fp / max(1, fp + tn),
             n_test=int(len(y)), n_attack=int(y.sum()))
    m["roc_auc"] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None
    return m


def record(experiment, model, train_ds, test_ds, m, setting=""):
    RESULTS.append(dict(experiment=experiment, model=model, train_ds=train_ds, test_ds=test_ds, setting=setting,
                        metrics=m))
    short = {k: v for k, v in m.items() if isinstance(v, float)}
    log(f"{experiment:16s} {model:8s} {train_ds:>10s} -> {test_ds:<10s} {setting:10s} "
        + " ".join(f"{k}={v:.4f}" for k, v in list(short.items())[:5]))


class Part:
    """One dataset, rows sorted by (host, start time), with step features and the split."""

    def __init__(self, key: str, df: pd.DataFrame, feats: pd.DataFrame):
        keep = df["proto"].astype(str).str.lower().isin(["tcp", "udp"]).values
        df, feats = df[keep].reset_index(drop=True), feats[keep].reset_index(drop=True)
        host_code = pd.factorize(df["host"].astype(str))[0]
        order = np.lexsort((np.arange(len(df)), np.floor(df["ts"].values), host_code))
        df, feats = df.iloc[order].reset_index(drop=True), feats.iloc[order].reset_index(drop=True)
        self.key = key
        self.raw = feats[COMMON_FEATURES].values.astype(np.float32)
        self.host = df["host"].values
        self.ts = df["ts"].values.astype(np.float64)
        self.dst = df["dst"].values
        self.dport = df["dport"].values
        self.cat = df["category"].values
        self.label = df["label"].values
        self.y = (self.cat != "Benign").astype(np.int8)
        self.steps = sq.step_matrix(feats, self.host, self.ts, self.dst, self.dport)
        self.start = sq.host_starts(self.host)
        buckets = np.floor(self.ts / 600).astype(np.int64)
        ub = np.unique(buckets)
        r = np.random.default_rng(SEED).random(len(ub))
        role = dict(zip(ub, np.where(r < 0.7, 0, np.where(r < 0.8, 1, 2))))
        rr = np.vectorize(role.get)(buckets)
        self.tr, self.va, self.te = (np.where(rr == k)[0] for k in (0, 1, 2))
        log(f"{key}: {len(df):,} TCP/UDP flows, {len(np.unique(self.host)):,} hosts, {len(ub):,} 10-min buckets; "
            f"train {len(self.tr):,} / val {len(self.va):,} / test {len(self.te):,}; attacks {self.y.mean():.1%}")


def stack(parts: list[Part]):
    """Concatenate parts into one index space (start offsets keep windows inside each part)."""
    steps = np.concatenate([p.steps for p in parts])
    raw = np.concatenate([p.raw for p in parts])
    offs = np.cumsum([0] + [len(p.y) for p in parts[:-1]])
    start = np.concatenate([p.start + o for p, o in zip(parts, offs)])
    y = np.concatenate([p.y for p in parts])
    tr = np.concatenate([p.tr + o for p, o in zip(parts, offs)])
    va = np.concatenate([p.va + o for p, o in zip(parts, offs)])
    return steps, raw, start, y, tr, va, offs


def per_category(p: Part, idx: np.ndarray, score: np.ndarray, thr: float = 0.5) -> dict:
    out = {}
    for c in np.unique(p.cat[idx]):
        sel = p.cat[idx] == c
        out[str(c)] = {"n": int(sel.sum()), "detected": float((score[sel] >= thr).mean())}
    return out


def evasion(model: sq.SeqLSTM, xgb, p: Part, idx: np.ndarray, eps_list=(0.0, 0.25, 0.5, 1.0, 2.0), steps=4,
            name: str = "LSTM"):
    """Adaptive black-box evasion. The attacker controls every one of THEIR flows in the window: pads, slows
    (longer flows AND longer gaps between them) and injects dummy packets. Victim flows are untouched."""
    out = []
    raw_df_last = pd.DataFrame(p.raw[idx], columns=COMMON_FEATURES)
    base_steps = sq.gather(p.steps, p.start, idx)            # zero-padded before the host's first flow
    for eps in eps_list:
        best_seq = sq.predict(model, p.steps, p.start, idx)
        best_flow = xgb.predict_proba(to_model_space(p.raw[idx]))[:, 1]
        if eps > 0:
            for pad, slow, inj in adv.grid(eps, steps):
                win = perturb_windows(base_steps, p.raw, p.y, p.start, idx, pad, slow, inj)
                best_seq = np.minimum(best_seq, _predict_windows(model, win))
                last = adv.perturb(raw_df_last, pad, slow, inj).values      # the flow being judged, perturbed
                pf = xgb.predict_proba(to_model_space(last))[:, 1]
                best_flow = np.minimum(best_flow, pf)
        out.append((eps, float((best_seq >= 0.5).mean()), float((best_flow >= 0.5).mean())))
        log(f"   evasion eps={eps}: {name} detects {out[-1][1]:.3f}, per-flow XGBoost {out[-1][2]:.3f}")
    return out


@torch.no_grad()
def _predict_windows(model, win, batch=8192):
    model.eval()
    dev = next(model.parameters()).device
    ps = [torch.sigmoid(model(torch.from_numpy(win[i:i + batch]).to(dev))).cpu().numpy()
          for i in range(0, len(win), batch)]
    return np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--out", default=str(MODELS_DIR / "sequence_bundle.pt"), help="where to write the live bundle")
    ap.add_argument("--live", default="LSTM-NoTiming-AdvAdaptive", choices=list(VARIANTS), help="variant saved as the live bundle")
    ap.add_argument("--save-variants", default=None, help="also save every variant's combined model to this folder")
    ap.add_argument("--push", action="store_true",
                    help="only push the saved sequence_results.json to the database (after a --no-db run)")
    args = ap.parse_args()
    if args.push:
        saved = json.loads((REPORTS_DIR / "sequence_results.json").read_text())
        push(saved["results"], saved["notes"])
        return

    log("loading CIC-IDS2017 (DistriNet re-labelled) and UNSW-NB15 full records")
    cdf = load_cicids2017_full()
    udf = load_unswnb15_full()
    parts = {"cic": Part("cic", cdf, from_cicids2017(cdf)), "unsw": Part("unsw", udf, from_unswnb15(udf))}
    del cdf, udf
    dev = sq.device()
    models, flow_models = {}, {}

    # ------------------------------------------------ train LSTM + per-flow XGBoost per dataset and combined
    for key in ("cic", "unsw", "both"):
        ps = [parts["cic"], parts["unsw"]] if key == "both" else [parts[key]]
        steps, raw, start, y, tr, va, _ = stack(ps)
        for v, drop in VARIANTS.items():
            log(f"== training {v} on {NAMES[key]} ({len(tr):,} windows, {args.epochs} epochs, {dev})")
            aug = {"random": adv_augmenter, "adaptive": adaptive_augmenter}[ADVERSARIAL[v]](raw, y, start) \
                if v in ADVERSARIAL else None
            models[(v, key)] = sq.train(steps, start, y, tr, va, epochs=args.epochs, log=log, drop=drop,
                                        augment=aug).to(dev)
        Xtr = to_model_space(np.concatenate([p.raw[p.tr] for p in ps]))
        ytr = np.concatenate([p.y[p.tr] for p in ps])
        log(f"== training per-flow XGBoost on {NAMES[key]} (same rows)")
        flow_models[key] = make_xgb(2).fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
        del steps, raw, Xtr

    # ------------------------------------------------ in-dataset + cross-dataset evaluation on test buckets
    for tk in ("cic", "unsw", "both"):
        for ek in ("cic", "unsw"):
            p = parts[ek]
            scores = {v: sq.predict(models[(v, tk)], p.steps, p.start, p.te) for v in VARIANTS}
            scores["XGBoost"] = flow_models[tk].predict_proba(to_model_space(p.raw[p.te]))[:, 1]
            exp = "seq_in_dataset" if tk == ek else ("seq_cross_dataset" if tk != "both" else "seq_combined")
            for name, s in scores.items():
                m = metrics(p.y[p.te], s)
                if tk == ek or tk == "both":
                    m["per_category"] = per_category(p, p.te, s)
                record(exp, name, NAMES[tk], NAMES[ek], m, "single flow" if name == "XGBoost" else "per-host sequence")

    # ------------------------------------------------ evasion (combined models, attack test windows)
    log("== adaptive evasion: combined LSTM vs per-flow XGBoost")
    for ek in ("cic", "unsw"):
        p = parts[ek]
        att = p.te[p.y[p.te] == 1]
        sample = RNG.choice(att, size=min(len(att), 8000), replace=False)
        for k, v in enumerate(VARIANTS):
            for eps, d_seq, d_flow in evasion(models[(v, "both")], flow_models["both"], p, sample, name=v):
                record("seq_robustness", v, "combined", NAMES[ek], {"detection_rate": d_seq, "eps": eps,
                                                                   "n_attack_flows": len(sample)}, f"eps={eps}")
                if k == 0:
                    record("seq_robustness", "XGBoost", "combined", NAMES[ek],
                           {"detection_rate": d_flow, "eps": eps, "n_attack_flows": len(sample)}, f"eps={eps}")

    # ------------------------------------------------ live bundle
    bg_parts = []
    for p in parts.values():
        ben = p.tr[p.y[p.tr] == 0]
        bg_parts.append(sq.gather(p.steps, p.start, RNG.choice(ben, size=60, replace=False)))
    version = time.strftime("seq-1.2.0+%Y%m%d%H%M", time.gmtime())

    def save(variant: str, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": version, "variant": variant, "drop": VARIANTS[variant],
                    "adversarial": ADVERSARIAL.get(variant),
                    "state_dict": models[(variant, "both")].cpu().eval().state_dict(), "n_feat": len(sq.STEP_FEATURES),
                    "step_features": sq.STEP_FEATURES, "seq_len": sq.SEQ_LEN, "alert_threshold": 0.9,
                    "background": np.concatenate(bg_parts).astype(np.float32)}, path)

    save(args.live, args.out)
    log(f"saved live sequence bundle {version} ({args.live})")
    if args.save_variants:
        for v in VARIANTS:
            save(v, Path(args.save_variants) / f"sequence_{v}.pt")
        log(f"saved all variants to {args.save_variants}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    notes = {"version": version, "live_variant": args.live, "seconds": int(time.time() - T0), "epochs": args.epochs, "seq_len": sq.SEQ_LEN,
             **{f"{k}_{s}": int(len(getattr(p, s))) for k, p in parts.items() for s in ("tr", "va", "te")}}
    (REPORTS_DIR / "sequence_results.json").write_text(json.dumps({"notes": notes, "results": RESULTS}, indent=1,
                                                                   default=float))
    if not args.no_db:
        push(RESULTS, notes)
    log("done")


def push(results: list[dict], notes: dict) -> None:
    """Replace the sequence results of the latest model run in the database."""
    with db.connect(database_url()) as conn:
        rid = conn.execute("select id from safenet.model_runs order by created_at desc limit 1").fetchone()[0]
        conn.execute("delete from safenet.eval_results where run_id=%s and experiment like 'seq_%%'", (rid,))
        for r in results:
            db.add_eval(conn, rid, r["experiment"], r["model"], r["train_ds"], r["test_ds"], r["metrics"],
                        r["setting"])
        conn.execute("update safenet.model_runs set notes = notes || jsonb_build_object('sequence', %s::jsonb) "
                     "where id=%s", (json.dumps(notes), rid))
        conn.commit()
    log(f"pushed {len(results)} sequence results to run {rid} ({socket.gethostname()})")


if __name__ == "__main__":
    main()
