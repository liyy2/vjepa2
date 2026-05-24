#!/usr/bin/env python
"""Can V-JEPA cached features predict the kinematic baseline's input features?

This is a diagnostic to test whether the rhythm signal is even RECOVERABLE from
V-JEPA cached embeddings. If yes, the bottleneck is decoding; if no, the
encoder fundamentally lacks the signal.

Pipeline:
  1. Load V-JEPA cache (train+val).
  2. Build per-clip feature vector: temporal-mean + temporal-std + (last-first) of cached [T, D] sequence.
  3. Fit Ridge regression on train → kinematic scalars (target).
  4. Report per-feature R² on val.
  5. Feed predicted kinematic features into the kinematic LightGBM head (rebuilt) and report final QWK.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score,
)


KIN_FEATS_DEFAULT = (
    "bbox_area_peak_rate_hz,"
    "bbox_area_peak_interval_cv,"
    "bbox_area_decrement,"
    "bbox_area_cycle_amplitude_mean,"
    "bbox_area_cycle_amplitude_std,"
    "bbox_area_dominant_freq_hz,"
    "bbox_area_bandpower_0p5_2_hz,"
    "bbox_area_bandpower_2_5_hz,"
    "bbox_area_bandpower_5_8_hz,"
    "bbox_area_velocity_mean_abs,"
    "bbox_area_velocity_std,"
    "bbox_area_peak_count,"
    "bbox_area_peak_prominence_mean,"
    "bbox_area_slope"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vjepa-train", required=True)
    p.add_argument("--vjepa-val", required=True)
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--features", default=KIN_FEATS_DEFAULT)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--out-json", default="")
    return p.parse_args()


def load_cache(path: str):
    d = np.load(path, allow_pickle=False)
    x = d["x"].astype(np.float32)  # [N, T, D]
    y = d["y"].astype(np.int64)
    mask = d["mask"].astype(bool) if "mask" in d.files else np.ones(x.shape[:2], dtype=bool)
    cov = [json.loads(str(s)) for s in d["coverage"]] if "coverage" in d.files else None
    return x, y, mask, cov


def cache_summary_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reduce [N, T, D] cache to a per-clip vector."""
    valid = mask.astype(np.float32)[..., None]
    cnt = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)[..., None]
    mean = (x * valid).sum(axis=1) / cnt[..., 0]
    var = (((x - mean[:, None]) ** 2) * valid).sum(axis=1) / cnt[..., 0]
    std = np.sqrt(np.clip(var, 0, None))
    # Velocity (Δ-magnitude mean)
    delta = x[:, 1:] - x[:, :-1]
    dmask = (mask[:, 1:] & mask[:, :-1]).astype(np.float32)[..., None]
    dcnt = (mask[:, 1:] & mask[:, :-1]).sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)[..., None]
    vmean = (np.abs(delta) * dmask).sum(axis=1) / dcnt[..., 0]
    # First and last vector
    first = x[:, 0]
    last = x[:, -1]
    return np.concatenate([mean, std, vmean, first, last, last - first], axis=1)


def load_kin_targets(fold_csv: str, kin_csv: str, feature_cols: list[str],
                     coverage: list[dict] | None) -> tuple[np.ndarray, np.ndarray]:
    fold_rows = list(csv.DictReader(open(fold_csv)))
    kin_rows = list(csv.DictReader(open(kin_csv)))
    by_clip = {r.get("clip_path"): r for r in kin_rows if r.get("clip_path")}
    if coverage is None:
        sample_indices = list(range(len(fold_rows)))
    else:
        sample_indices = [int(round(float(c.get("sample_index", -1)))) for c in coverage]
    N = len(sample_indices)
    Y = np.full((N, len(feature_cols)), np.nan, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    for i, si in enumerate(sample_indices):
        if si < 0 or si >= len(fold_rows):
            continue
        clip = fold_rows[si].get("clip_path")
        kin = by_clip.get(clip)
        if kin is None:
            continue
        vals = []
        ok = True
        for col in feature_cols:
            try:
                v = float(kin.get(col, ""))
                if not np.isfinite(v):
                    ok = False; break
                vals.append(v)
            except Exception:
                ok = False; break
        if ok:
            Y[i] = vals
            mask[i] = True
    return Y, mask


def main():
    args = parse_args()
    feats = [c.strip() for c in args.features.split(",") if c.strip()]

    # Load V-JEPA caches and labels
    x_tr, y_tr, m_tr, cov_tr = load_cache(args.vjepa_train)
    x_va, y_va, m_va, cov_va = load_cache(args.vjepa_val)

    # Compute per-clip summary features
    F_tr = cache_summary_features(x_tr, m_tr)
    F_va = cache_summary_features(x_va, m_va)
    print(f"V-JEPA summary feature dim: {F_tr.shape[1]}")

    # Load kinematic targets (aligned to cache order via coverage.sample_index)
    Y_tr, Mtr = load_kin_targets(args.fold_train_csv, args.kin_train_csv, feats, cov_tr)
    Y_va, Mva = load_kin_targets(args.fold_val_csv, args.kin_val_csv, feats, cov_va)
    print(f"train valid rows: {Mtr.sum()}/{len(Mtr)}; val valid rows: {Mva.sum()}/{len(Mva)}")

    # Standardize targets
    mu = np.nanmean(Y_tr[Mtr], axis=0)
    sd = np.nanstd(Y_tr[Mtr], axis=0) + 1e-6
    Yz_tr = (Y_tr[Mtr] - mu) / sd
    Yz_va = (Y_va[Mva] - mu) / sd

    # Fit per-feature ridge regression
    print(f"\n{'feature':<40} {'R²':>8}")
    print("-" * 50)
    r2_per = []
    Y_pred_va = np.zeros_like(Yz_va)
    for j, name in enumerate(feats):
        reg = Ridge(alpha=args.ridge_alpha)
        reg.fit(F_tr[Mtr], Yz_tr[:, j])
        pred = reg.predict(F_va[Mva])
        r2 = r2_score(Yz_va[:, j], pred)
        print(f"{name:<40} {r2:>8.4f}")
        r2_per.append(r2)
        Y_pred_va[:, j] = pred

    print(f"\nMEAN R² over {len(feats)} features: {np.mean(r2_per):.4f}")
    print(f"MEDIAN R²: {np.median(r2_per):.4f}")

    # Also unstandardize and feed predicted kin features through a fresh LightGBM
    # → see if predicted-kin can drive the kinematic baseline.
    Y_pred_unstd = Y_pred_va * sd + mu
    Y_true_va = Y_va[Mva]

    # Now train LightGBM on TRAIN kinematic features (full 14 features) and predict on val
    try:
        import lightgbm as lgb
        params = dict(
            objective="multiclass", num_class=5, metric="multi_logloss",
            num_leaves=15, learning_rate=0.05, feature_fraction=0.7,
            bagging_fraction=0.85, bagging_freq=3, seed=0, verbose=-1,
        )

        # Baseline LightGBM: train on TRUE kinematic features, evaluate on TRUE val features
        tr_set = lgb.Dataset(Y_tr[Mtr], label=y_tr[Mtr].astype(np.float64))
        va_set_true = lgb.Dataset(Y_va[Mva], label=y_va[Mva].astype(np.float64), reference=tr_set)
        model = lgb.train(params, tr_set, num_boost_round=200, valid_sets=[va_set_true],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        pred_true = model.predict(Y_va[Mva]).argmax(1)
        qwk_true = cohen_kappa_score(y_va[Mva], pred_true, weights="quadratic")
        print(f"\nLightGBM with TRUE kin features (val): qwk={qwk_true:.4f} acc={(pred_true==y_va[Mva]).mean():.4f}")

        # Now predict val using LightGBM trained on TRUE features but with PREDICTED val features
        pred_pred = model.predict(Y_pred_unstd).argmax(1)
        qwk_pred = cohen_kappa_score(y_va[Mva], pred_pred, weights="quadratic")
        print(f"LightGBM with V-JEPA-predicted kin features (val): qwk={qwk_pred:.4f} acc={(pred_pred==y_va[Mva]).mean():.4f}")
    except ImportError:
        print("lightgbm not available")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "features": feats,
            "r2_per_feature": [float(r) for r in r2_per],
            "r2_mean": float(np.mean(r2_per)),
            "r2_median": float(np.median(r2_per)),
        }, indent=2))


if __name__ == "__main__":
    main()
