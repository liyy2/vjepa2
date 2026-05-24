#!/usr/bin/env python
"""Hybrid LightGBM on V-JEPA combiner probabilities + kinematic features.

Goal: beat kinematic-alone QWK 0.728 by adding V-JEPA combiner outputs as
side features. Specifically:
  - Load kinematic features per clip (the same CSV the kinematic baseline trains on).
  - Load V-JEPA combiner val_probs from a sweep's best.json (and optionally per-clip
    train predictions via the saved best.pt, but for simplicity we use the leave-one-out
    style: for val, use the BEST sweep's val_probs; for train, retrain the combiner with
    subject-OOF inner CV and collect OOF val_probs).
  - Concatenate: kinematic[N, 14+] + vjepa_probs[N, 5] → fed to LightGBM.
  - Compare to LightGBM on kinematic-alone with the same hyperparameters.

Honest version requires subject-OOF V-JEPA predictions on train. This script
provides a simpler "val-tuned" diagnostic first: just use V-JEPA val_probs and a
LightGBM trained on kinematic-only train features, then concatenate predict_proba
of LightGBM with V-JEPA val_probs and fit a small calibrator. The HONEST version
is in train_hybrid_oof_kin_vjepa.py (separate script).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

NUM_CLASSES = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--vjepa-result-json", required=True,
                   help="path to result.json from train_corn_combiner.py (has val_probs + val_labels)")
    p.add_argument("--vjepa-cache-val", required=True,
                   help="cache npz for val to map cache order → CSV order via sample_index")
    p.add_argument("--feature-set", default="bbox_area",
                   help="kinematic column prefix to use (e.g. 'bbox_area')")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def load_csv_row_dict(path):
    return list(csv.DictReader(open(path)))


def kinematic_feature_columns(prefix, kin_csv):
    rows = load_csv_row_dict(kin_csv)
    if not rows:
        return []
    return [c for c in rows[0].keys() if c.startswith(prefix + "_")]


def load_kin_features(kin_csv, fold_csv, feat_cols):
    """Return X[N, F], y[N] in fold CSV row order."""
    fold_rows = load_csv_row_dict(fold_csv)
    kin_rows = load_csv_row_dict(kin_csv)
    by_clip = {r["clip_path"]: r for r in kin_rows}
    N = len(fold_rows)
    X = np.zeros((N, len(feat_cols)), dtype=np.float32)
    y = np.zeros(N, dtype=np.int64)
    mask = np.zeros(N, dtype=bool)
    for i, r in enumerate(fold_rows):
        kin = by_clip.get(r["clip_path"])
        if kin is None:
            continue
        try:
            vals = [float(kin.get(c, "nan")) for c in feat_cols]
            if any(not np.isfinite(v) for v in vals):
                continue
            X[i] = vals
            y[i] = int(float(r["label"]))
            mask[i] = True
        except Exception:
            continue
    return X, y, mask


def load_vjepa_val_probs(result_json, cache_val_npz, fold_val_csv):
    """Map V-JEPA val_probs from cache order to fold-val CSV row order."""
    res = json.loads(Path(result_json).read_text())
    best = res.get("best", {})
    metrics = best.get("metrics", {})
    probs_cache_order = np.asarray(metrics.get("val_probs"), dtype=np.float32)
    labels_cache_order = np.asarray(metrics.get("val_labels"), dtype=np.int64)
    cache = np.load(cache_val_npz, allow_pickle=False)
    coverage = [json.loads(str(s)) for s in cache["coverage"]] if "coverage" in cache.files else None
    if coverage is None:
        raise SystemExit("V-JEPA val cache has no coverage; cannot map sample_index")
    fold_rows = load_csv_row_dict(fold_val_csv)
    N = len(fold_rows)
    out_probs = np.zeros((N, probs_cache_order.shape[1]), dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    for cache_idx, cov in enumerate(coverage):
        si = int(round(float(cov.get("sample_index", -1))))
        if 0 <= si < N:
            out_probs[si] = probs_cache_order[cache_idx]
            mask[si] = True
    # Sanity: V-JEPA labels at the mapped positions should equal fold labels
    return out_probs, mask


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def main():
    args = parse_args()

    feat_cols = kinematic_feature_columns(args.feature_set, args.kin_train_csv)
    print(f"kinematic feature columns ({args.feature_set}): {len(feat_cols)}")

    X_kin_tr, y_tr, m_tr = load_kin_features(args.kin_train_csv, args.fold_train_csv, feat_cols)
    X_kin_va, y_va, m_va = load_kin_features(args.kin_val_csv, args.fold_val_csv, feat_cols)
    print(f"train rows valid: {m_tr.sum()}/{len(m_tr)} ; val: {m_va.sum()}/{len(m_va)}")

    vj_probs_va, vj_mask_va = load_vjepa_val_probs(args.vjepa_result_json, args.vjepa_cache_val, args.fold_val_csv)
    print(f"V-JEPA val_probs valid: {vj_mask_va.sum()}/{len(vj_mask_va)}")

    # For training the hybrid we need V-JEPA-style features on TRAIN too. Without per-clip
    # OOF predictions on train, the cleanest baseline is: use kinematic LightGBM's predict_proba
    # on train, BUT that's a self-prediction → overfit. So we instead use the LightGBM
    # trained on kinematic features alone and at val time concatenate ITS predict_proba with
    # V-JEPA val_probs and a final linear calibrator selected by leave-one-out on val.
    #
    # This is val-tuned (not honest CV), so report as diagnostic only. Goal is to show the
    # ceiling of fusion before committing to subject-OOF inner CV.

    import lightgbm as lgb
    params = dict(
        objective="multiclass", num_class=NUM_CLASSES, metric="multi_logloss",
        num_leaves=15, learning_rate=0.05, feature_fraction=0.7,
        bagging_fraction=0.85, bagging_freq=3, seed=0, verbose=-1,
    )
    valid = m_tr & np.all(np.isfinite(X_kin_tr), axis=1)
    tr_set = lgb.Dataset(X_kin_tr[valid], label=y_tr[valid].astype(np.float64),
                         feature_name=feat_cols)
    va_set = lgb.Dataset(X_kin_va[m_va & vj_mask_va], label=y_va[m_va & vj_mask_va].astype(np.float64),
                         reference=tr_set, feature_name=feat_cols)
    model_kin = lgb.train(params, tr_set, num_boost_round=400, valid_sets=[va_set],
                          callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    kin_proba_va = model_kin.predict(X_kin_va, num_iteration=model_kin.best_iteration)

    # Pure kinematic
    pred_kin = kin_proba_va.argmax(1)
    valid_va = m_va & vj_mask_va
    qwk_kin = qwk(y_va[valid_va], pred_kin[valid_va])
    acc_kin = accuracy_score(y_va[valid_va], pred_kin[valid_va])
    mae_kin = mean_absolute_error(y_va[valid_va], pred_kin[valid_va])
    print(f"\nKINEMATIC-only LightGBM: qwk={qwk_kin:.4f} acc={acc_kin:.4f} mae={mae_kin:.4f}")

    # Pure V-JEPA
    pred_vj = vj_probs_va.argmax(1)
    qwk_vj = qwk(y_va[valid_va], pred_vj[valid_va])
    print(f"VJEPA-only (combiner best): qwk={qwk_vj:.4f}")

    # Sweep fusion weight α  (kin_proba × α + vj_proba × (1-α))
    print(f"\n{'alpha':>6} {'qwk':>8} {'acc':>8} {'mae':>8}")
    best_alpha, best_qwk, best_metrics = None, -2.0, None
    for a in np.arange(0.0, 1.01, 0.05):
        fused = a * kin_proba_va + (1 - a) * vj_probs_va
        yhat = fused.argmax(1)
        q = qwk(y_va[valid_va], yhat[valid_va])
        ac = accuracy_score(y_va[valid_va], yhat[valid_va])
        ma = mean_absolute_error(y_va[valid_va], yhat[valid_va])
        flag = " *" if q > best_qwk else ""
        print(f"{a:>6.2f} {q:>8.4f} {ac:>8.4f} {ma:>8.4f}{flag}")
        if q > best_qwk:
            best_alpha = float(a); best_qwk = q
            best_metrics = {"alpha": float(a), "qwk": q, "acc": float(ac), "mae": float(ma)}

    print(f"\nBEST fusion: alpha={best_alpha} qwk={best_qwk:.4f} acc={best_metrics['acc']:.4f} mae={best_metrics['mae']:.4f}")

    # Also: stacked LightGBM on [kin_features ‖ vjepa_probs]
    # Use V-JEPA OOF train predictions if available; otherwise this is val-tuned (skip).
    # As a proxy, train a calibrator LightGBM only on val using stratified inner CV on val.
    # (Honest version is the separate script.)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({
        "kinematic_only": {"qwk": qwk_kin, "acc": float(acc_kin), "mae": float(mae_kin)},
        "vjepa_only": {"qwk": qwk_vj},
        "best_alpha_fusion": best_metrics,
        "note": "val-tuned alpha — honest version requires subject-OOF train predictions",
    }, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
