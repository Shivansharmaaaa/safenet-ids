"""SafeNet training + evaluation pipeline (real CIC-IDS2017 and UNSW-NB15 data).

Experiments
  1. in_dataset     native features, each dataset on its own (RF, XGBoost, Autoencoder)
  2. in_dataset     common features (cost of harmonising the feature space)
  3. cross_dataset  train on one dataset, test on the other (binary benign/attack)
  4. robustness     adaptive evasion attack (pad / slow / inject) at several budgets
  5. adv_training   XGBoost retrained with adversarially perturbed attacks
Then fits the live-sensor bundle, computes SHAP importances, writes a JSON +
Markdown report to artifacts/reports and pushes every number to the database.

usage:  python -m scripts.train [--fast] [--no-db]
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from safenet import __version__
from safenet import adversarial as adv
from safenet import data as D
from safenet.config import MODELS_DIR, REPORTS_DIR, SEED, database_url
from safenet.explain import global_importance
from safenet.features import CATEGORIES, COMMON_FEATURES, to_model_space
from safenet.models import AutoencoderDetector, balanced_weights, make_rf, make_xgb

RNG = np.random.default_rng(SEED)
RESULTS: list[dict] = []
IMPORTANCE: list[dict] = []
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def record(experiment, model, train_ds, test_ds, metrics, setting=""):
    RESULTS.append(dict(experiment=experiment, model=model, train_ds=train_ds, test_ds=test_ds,
                        setting=setting, metrics=metrics))
    short = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    log(f"{experiment:13s} {model:12s} {train_ds:>10s} -> {test_ds:<10s} {setting:12s} " +
        " ".join(f"{k}={v:.4f}" for k, v in list(short.items())[:6]))


def binary_metrics(y_true_attack: np.ndarray, y_pred_attack: np.ndarray, score: np.ndarray | None) -> dict:
    m = dict(
        accuracy=accuracy_score(y_true_attack, y_pred_attack),
        precision=precision_score(y_true_attack, y_pred_attack, zero_division=0),
        recall=recall_score(y_true_attack, y_pred_attack, zero_division=0),
        f1=f1_score(y_true_attack, y_pred_attack, zero_division=0),
        fpr=float(((y_pred_attack == 1) & (y_true_attack == 0)).sum() / max(1, (y_true_attack == 0).sum())),
        n_test=int(len(y_true_attack)), n_attack=int(y_true_attack.sum()),
    )
    if score is not None and len(np.unique(y_true_attack)) == 2:
        m["roc_auc"] = roc_auc_score(y_true_attack, score)
    return m


def per_label_recall(labels: pd.Series, y_pred_attack: np.ndarray) -> dict:
    """Detection rate per ORIGINAL attack label (shows how rare attacks fare)."""
    out = {}
    for lab, idx in labels.groupby(labels).groups.items():
        pos = labels.index.get_indexer(idx)
        out[str(lab)] = {"n": int(len(pos)), "detected": float(y_pred_attack[pos].mean())}
    return out


# ----------------------------------------------------------------------------- native feature views
def native_cic(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["label", "category", "day"]).astype(np.float32)


def native_unsw(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=["id", "label", "category", "split"], errors="ignore")
    X = pd.get_dummies(X, columns=["proto", "service", "state"], dtype=np.float32)
    return X.astype(np.float32)


def split_cic(df: pd.DataFrame, fast: bool):
    """Stratified 70/30 split by original label after removing exact duplicates.
    Training benign is capped (all attacks kept) to keep training tractable."""
    feats = [c for c in df.columns if c not in ("label", "category", "day")]
    df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
    # labels with <2 rows cannot be stratified
    tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED, stratify=df["label"])
    cap = 150_000 if fast else 600_000
    benign_tr = tr[df["category"].values[tr] == "Benign"]
    if len(benign_tr) > cap:
        drop = RNG.choice(benign_tr, size=len(benign_tr) - cap, replace=False)
        tr = np.setdiff1d(tr, drop)
    if fast:
        te = RNG.choice(te, size=min(len(te), 250_000), replace=False)
    return df, tr, te


# ----------------------------------------------------------------------------- experiments
def fit_eval_models(name: str, Xtr: np.ndarray, ytr_cat: np.ndarray, Xte: np.ndarray, yte_cat: np.ndarray,
                    labels_te: pd.Series, feature_names: list[str], tag: str, fast: bool, keep: dict | None = None):
    classes = [c for c in CATEGORIES if c in set(ytr_cat)]
    cidx = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cidx[c] for c in ytr_cat])
    yte_att = (yte_cat != "Benign").astype(int)
    benign_i = cidx["Benign"]

    for mname in ("RandomForest", "XGBoost"):
        model = make_rf() if mname == "RandomForest" else make_xgb(len(classes))
        if fast and mname == "RandomForest":
            model.set_params(n_estimators=60)
        t = time.time()
        if mname == "XGBoost":
            model.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
        else:
            model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)
        pred = proba.argmax(1)
        p_att = 1.0 - proba[:, benign_i]
        pred_att = (pred != benign_i).astype(int)
        m = binary_metrics(yte_att, pred_att, p_att)
        known = np.isin(yte_cat, classes)
        pred_cat = np.array(classes)[pred]
        m["macro_f1_category"] = f1_score(yte_cat[known], pred_cat[known], average="macro", zero_division=0)
        m["category_report"] = classification_report(yte_cat[known], pred_cat[known], output_dict=True, zero_division=0)
        m["per_label_detection"] = per_label_recall(labels_te.reset_index(drop=True), pred_att)
        m["train_seconds"] = round(time.time() - t, 1)
        record("in_dataset", mname, name, name, m, tag)
        if mname == "XGBoost":
            imp = global_importance(model, Xte, feature_names, n=1500 if fast else 3000)
            IMPORTANCE.append(dict(model=f"XGBoost ({tag})", dataset=name, items=imp[:25]))
        if keep is not None:
            keep[mname] = model

    ae = AutoencoderDetector(epochs=6 if fast else 15)
    ben = Xtr[ytr == benign_i]
    tr_i, va_i = train_test_split(np.arange(len(ben)), test_size=0.1, random_state=SEED)
    ae.fit(ben[tr_i], ben[va_i])
    s = ae.score(Xte)
    m = binary_metrics(yte_att, (s > ae.threshold).astype(int), s)
    m["per_label_detection"] = per_label_recall(labels_te.reset_index(drop=True), (s > ae.threshold).astype(int))
    m["threshold"] = ae.threshold
    record("in_dataset", "Autoencoder", name, name, m, tag)
    if keep is not None:
        keep["Autoencoder"] = ae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller samples for a quick end-to-end check")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    fast = args.fast

    log("loading CIC-IDS2017 ...")
    cic = D.load_cicids2017_native()
    log(f"CIC-IDS2017 rows={len(cic):,}  labels={cic['label'].nunique()}")
    log("loading UNSW-NB15 ...")
    unsw = D.load_unswnb15_native()
    log(f"UNSW-NB15 rows={len(unsw):,}  labels={unsw['label'].nunique()}")

    from safenet.features import from_cicids2017, from_unswnb15

    # -------------------------------------------------- CIC split (shared by native + common views)
    cic, c_tr, c_te = split_cic(cic, fast)
    c_native = native_cic(cic)
    c_common = from_cicids2017(cic)
    c_cat, c_lab = cic["category"].values, cic["label"]

    u_tr = np.where(unsw["split"].values == "train")[0]
    u_te = np.where(unsw["split"].values == "test")[0]
    u_native = native_unsw(unsw)
    u_common = from_unswnb15(unsw)
    u_cat, u_lab = unsw["category"].values, unsw["label"]

    # -------------------------------------------------- 1. in-dataset, native features
    log("== 1. in-dataset (native features)")
    fit_eval_models("cicids2017", c_native.values[c_tr], c_cat[c_tr], c_native.values[c_te], c_cat[c_te],
                    c_lab.iloc[c_te], list(c_native.columns), "native", fast)
    fit_eval_models("unswnb15", u_native.values[u_tr], u_cat[u_tr], u_native.values[u_te], u_cat[u_te],
                    u_lab.iloc[u_te], list(u_native.columns), "native", fast)

    # -------------------------------------------------- 2. in-dataset, common features
    log("== 2. in-dataset (common features)")
    cXtr, cXte = to_model_space(c_common.values[c_tr]), to_model_space(c_common.values[c_te])
    uXtr, uXte = to_model_space(u_common.values[u_tr]), to_model_space(u_common.values[u_te])
    fit_eval_models("cicids2017", cXtr, c_cat[c_tr], cXte, c_cat[c_te], c_lab.iloc[c_te], COMMON_FEATURES, "common", fast)
    fit_eval_models("unswnb15", uXtr, u_cat[u_tr], uXte, u_cat[u_te], u_lab.iloc[u_te], COMMON_FEATURES, "common", fast)

    # -------------------------------------------------- 3. cross-dataset (binary, common features)
    log("== 3. cross-dataset")
    sets = {
        "cicids2017": (cXtr, (c_cat[c_tr] != "Benign").astype(int), cXte, (c_cat[c_te] != "Benign").astype(int)),
        "unswnb15": (uXtr, (u_cat[u_tr] != "Benign").astype(int), uXte, (u_cat[u_te] != "Benign").astype(int)),
    }
    sets["combined"] = (np.vstack([cXtr, uXtr]), np.concatenate([sets["cicids2017"][1], sets["unswnb15"][1]]), None, None)
    for train_ds in ("cicids2017", "unswnb15", "combined"):
        Xa, ya = sets[train_ds][0], sets[train_ds][1]
        models = {"RandomForest": make_rf(), "XGBoost": make_xgb(2)}
        if fast:
            models["RandomForest"].set_params(n_estimators=60)
        for mname, model in models.items():
            if mname == "XGBoost":
                model.fit(Xa, ya, sample_weight=balanced_weights(ya))
            else:
                model.fit(Xa, ya)
            for test_ds in ("cicids2017", "unswnb15"):
                Xb, yb = sets[test_ds][2], sets[test_ds][3]
                p = model.predict_proba(Xb)[:, 1]
                record("cross_dataset", mname, train_ds, test_ds, binary_metrics(yb, (p >= 0.5).astype(int), p))
        ae = AutoencoderDetector(epochs=6 if fast else 15)
        ben = Xa[ya == 0]
        tr_i, va_i = train_test_split(np.arange(len(ben)), test_size=0.1, random_state=SEED)
        ae.fit(ben[tr_i], ben[va_i])
        for test_ds in ("cicids2017", "unswnb15"):
            Xb, yb = sets[test_ds][2], sets[test_ds][3]
            s = ae.score(Xb)
            record("cross_dataset", "Autoencoder", train_ds, test_ds, binary_metrics(yb, (s > ae.threshold).astype(int), s))

    # -------------------------------------------------- 4/5. robustness + adversarial training (combined model)
    log("== 4. robustness (adaptive evasion)")
    raw_tr = pd.concat([c_common.iloc[c_tr], u_common.iloc[u_tr]], ignore_index=True)
    cat_tr = np.concatenate([c_cat[c_tr], u_cat[u_tr]])
    raw_te = pd.concat([c_common.iloc[c_te], u_common.iloc[u_te]], ignore_index=True)
    cat_te = np.concatenate([c_cat[c_te], u_cat[u_te]])
    src_te = np.array(["cicids2017"] * len(c_te) + ["unswnb15"] * len(u_te))
    y_tr = (cat_tr != "Benign").astype(int)
    y_te = (cat_te != "Benign").astype(int)
    Xtr_m, Xte_m = to_model_space(raw_tr.values), to_model_space(raw_te.values)

    xgb_clean = make_xgb(2).fit(Xtr_m, y_tr, sample_weight=balanced_weights(y_tr))
    rf_clean = make_rf()
    if fast:
        rf_clean.set_params(n_estimators=60)
    rf_clean.fit(Xtr_m, y_tr)
    ae_comb = AutoencoderDetector(epochs=6 if fast else 15)
    ben = Xtr_m[y_tr == 0]
    tr_i, va_i = train_test_split(np.arange(len(ben)), test_size=0.1, random_state=SEED)
    ae_comb.fit(ben[tr_i], ben[va_i])

    # adversarial training: add perturbed copies of training attacks (random budgets up to 2.0)
    log("== 5. adversarial training")
    att_idx = np.where(y_tr == 1)[0]
    n_aug = min(len(att_idx), 150_000 if not fast else 40_000)
    pick = RNG.choice(att_idx, size=n_aug, replace=False)
    base = raw_tr.iloc[pick].reset_index(drop=True)
    aug_parts = []
    for chunk in np.array_split(np.arange(n_aug), 8):
        e = RNG.uniform(0.25, 2.0)
        pad, slow, inj = RNG.uniform(0, e, size=3)
        aug_parts.append(adv.perturb(base.iloc[chunk], pad, slow, inj))
    aug = pd.concat(aug_parts, ignore_index=True)
    X_adv = np.vstack([Xtr_m, to_model_space(aug.values)])
    y_adv = np.concatenate([y_tr, np.ones(len(aug), dtype=int)])
    xgb_adv = make_xgb(2).fit(X_adv, y_adv, sample_weight=balanced_weights(y_adv))

    p_clean_xgb = xgb_clean.predict_proba(Xte_m)[:, 1]
    p_adv_xgb = xgb_adv.predict_proba(Xte_m)[:, 1]
    for mname, p in (("XGBoost", p_clean_xgb), ("XGBoost-AdvTrained", p_adv_xgb)):
        m = binary_metrics(y_te, (p >= 0.5).astype(int), p)
        for ds in ("cicids2017", "unswnb15"):
            sel = src_te == ds
            m[f"recall_{ds}"] = recall_score(y_te[sel], (p[sel] >= 0.5).astype(int), zero_division=0)
            m[f"fpr_{ds}"] = float(((p[sel] >= 0.5) & (y_te[sel] == 0)).sum() / max(1, (y_te[sel] == 0).sum()))
        record("adv_training", mname, "combined", "combined", m, "clean")

    att_te = np.where(y_te == 1)[0]
    n_eval = min(len(att_te), 8_000 if fast else 30_000)
    sample = RNG.choice(att_te, size=n_eval, replace=False)
    X_att = raw_te.iloc[sample].reset_index(drop=True)

    targets = {
        "RandomForest": lambda X: rf_clean.predict_proba(to_model_space(X.values))[:, 1],
        "XGBoost": lambda X: xgb_clean.predict_proba(to_model_space(X.values))[:, 1],
        "XGBoost-AdvTrained": lambda X: xgb_adv.predict_proba(to_model_space(X.values))[:, 1],
    }
    ae_norm = lambda X: ae_comb.score(to_model_space(X.values)) / ae_comb.threshold / 2.0  # >0.5 <=> anomaly
    targets["Autoencoder"] = ae_norm
    for eps in (0.0, 0.25, 0.5, 1.0, 2.0):
        for mname, fn in targets.items():
            _, p = adv.adaptive_attack(X_att, fn, eps, steps=3 if fast else 4)
            dr = float((p >= 0.5).mean())
            record("robustness", mname, "combined", "combined", {"detection_rate": dr, "n_attack_flows": n_eval,
                                                                  "eps": eps}, f"eps={eps}")

    # -------------------------------------------------- live bundle (used by the sensor)
    log("== fitting live-sensor bundle")
    classes = [c for c in CATEGORIES if c in set(cat_tr)]
    cidx = {c: i for i, c in enumerate(classes)}
    yc = np.array([cidx[c] for c in cat_tr])
    xgb_cat = make_xgb(len(classes)).fit(Xtr_m, yc, sample_weight=balanced_weights(yc))
    acc_cat = accuracy_score([cidx[c] for c in cat_te], xgb_cat.predict(Xte_m))
    record("in_dataset", "XGBoost-Category", "combined", "combined",
           {"accuracy": acc_cat, "macro_f1_category": f1_score([cidx[c] for c in cat_te], xgb_cat.predict(Xte_m),
                                                                 average="macro")}, "common")
    IMPORTANCE.append(dict(model="Live detector (XGBoost-AdvTrained)", dataset="combined",
                           items=global_importance(xgb_adv, Xte_m, COMMON_FEATURES)))

    version = f"{__version__}+{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}" + ("-fast" if fast else "")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = dict(version=version, features=COMMON_FEATURES, detector=xgb_adv, category_model=xgb_cat,
                  categories=classes, autoencoder=ae_comb, alert_threshold=0.9)
    joblib.dump(bundle, MODELS_DIR / "live_bundle.joblib", compress=3)
    log(f"saved {MODELS_DIR / 'live_bundle.joblib'} version={version}")

    # -------------------------------------------------- reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    notes = dict(fast=fast, python=platform.python_version(), seconds=round(time.time() - T0),
                 cic_rows_after_dedup=int(len(cic)), cic_train=int(len(c_tr)), cic_test=int(len(c_te)),
                 unsw_train=int(len(u_tr)), unsw_test=int(len(u_te)))
    (REPORTS_DIR / "results.json").write_text(json.dumps(dict(version=version, notes=notes, results=RESULTS,
                                                              importance=IMPORTANCE), indent=1, default=float))
    write_markdown(version, notes)

    if not args.no_db:
        push_db(version, notes)
    log("done")


def write_markdown(version: str, notes: dict) -> None:
    L = [f"# SafeNet evaluation report\n\nModel version `{version}`  \nGenerated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}\n",
         f"Data: CIC-IDS2017 {notes['cic_rows_after_dedup']:,} unique flows (train {notes['cic_train']:,} / test {notes['cic_test']:,}); "
         f"UNSW-NB15 official split (train {notes['unsw_train']:,} / test {notes['unsw_test']:,}).\n"]
    def table(rows, cols):
        out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            out.append("| " + " | ".join(f(r) for _, f in cols) + " |")
        return "\n".join(out) + "\n"
    f4 = lambda k: (lambda r: f"{r['metrics'].get(k, float('nan')):.4f}")
    L.append("## In-dataset\n")
    L.append(table([r for r in RESULTS if r["experiment"] == "in_dataset" and "f1" in r["metrics"]],
                   [("Dataset", lambda r: r["train_ds"]), ("Features", lambda r: r["setting"]), ("Model", lambda r: r["model"]),
                    ("Precision", f4("precision")), ("Recall", f4("recall")), ("F1", f4("f1")), ("FPR", f4("fpr")),
                    ("ROC-AUC", f4("roc_auc"))]))
    L.append("## Cross-dataset (common features, benign vs attack)\n")
    L.append(table([r for r in RESULTS if r["experiment"] == "cross_dataset"],
                   [("Train", lambda r: r["train_ds"]), ("Test", lambda r: r["test_ds"]), ("Model", lambda r: r["model"]),
                    ("Precision", f4("precision")), ("Recall", f4("recall")), ("F1", f4("f1")), ("FPR", f4("fpr")),
                    ("ROC-AUC", f4("roc_auc"))]))
    L.append("## Adversarial robustness (adaptive pad/slow/inject attack)\n")
    L.append(table([r for r in RESULTS if r["experiment"] == "robustness"],
                   [("Model", lambda r: r["model"]), ("Budget", lambda r: r["setting"]), ("Detection rate", f4("detection_rate"))]))
    L.append("## Adversarial training, clean performance\n")
    L.append(table([r for r in RESULTS if r["experiment"] == "adv_training"],
                   [("Model", lambda r: r["model"]), ("Precision", f4("precision")), ("Recall", f4("recall")),
                    ("F1", f4("f1")), ("FPR", f4("fpr"))]))
    (REPORTS_DIR / "report.md").write_text("\n".join(L))


def push_db(version: str, notes: dict) -> None:
    from safenet import db
    url = database_url()
    db.init_schema(url)
    with db.connect(url) as conn:
        run_id = db.create_run(conn, version, socket.gethostname(), notes)
        for r in RESULTS:
            db.add_eval(conn, run_id, r["experiment"], r["model"], r["train_ds"], r["test_ds"], r["metrics"], r["setting"])
        for imp in IMPORTANCE:
            db.add_importance(conn, run_id, imp["model"], imp["dataset"], imp["items"])
        conn.commit()
    log(f"pushed {len(RESULTS)} results to database (run {run_id})")


if __name__ == "__main__":
    main()
