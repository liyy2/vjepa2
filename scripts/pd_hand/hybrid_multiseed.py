#!/usr/bin/env python
"""Multi-seed sanity check for the hybrid kin + V-JEPA result.

Trains ExtraTrees with many random_states on:
  (a) kin-only feature set (old_118)
  (b) kin + V-JEPA feature set (all)
  (c) kin + meta (old_openmeta)

Reports mean ± std QWK over seeds to see if the +0.038 lift is real or cherry-picked.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.pipeline import make_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--vjepa-prefix", default="vjepa")
    p.add_argument("--seeds", default="0,1,2,17,42,98,233,777,1234,2222",
                   help="comma-separated random_state values")
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--old-118-cols", default="",
                   help="Optional path to a JSON list of column names defining the old_118 feature set.")
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def split_features(rows, vjepa_prefix):
    """Return (kin_cols, vj_cols, meta_cols). Only requires the column to be present
    in row 0 (no float-test) so we don't exclude cols with empty first-row values."""
    keys = list(rows[0].keys())
    ignore = {"row_id", "subject_id", "visit_id", "side", "clip_path", "label",
              "side_right_orig"}
    candidate_keys = [k for k in keys if k not in ignore]
    vj_cols = [k for k in candidate_keys if k.startswith(vjepa_prefix)]
    meta_cols = [k for k in candidate_keys if k.startswith(("dx_", "item35"))
                 or k in {"side_right", "duration_s_manifest", "start_s_manifest"}]
    kin_cols = [k for k in candidate_keys if k not in vj_cols and k not in meta_cols]
    return kin_cols, vj_cols, meta_cols


def to_matrix(rows, cols):
    X = np.full((len(rows), len(cols)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            try:
                v = float(r.get(c, "nan"))
                if np.isfinite(v):
                    X[i, j] = v
            except Exception:
                pass
    return X


def fit_eval(X_tr, y_tr, X_va, y_va, seed, n_estimators):
    # Match `extra_trees_old_seed0` config from the kinematic baseline
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    )
    model.fit(X_tr, y_tr)
    p = model.predict(X_va).astype(np.int64)
    return {
        "qwk": float(cohen_kappa_score(y_va, p, weights="quadratic")),
        "acc": float(accuracy_score(y_va, p)),
        "mae": float(mean_absolute_error(y_va, p)),
    }


def main():
    args = parse_args()
    rows_tr = list(csv.DictReader(open(args.train_csv)))
    rows_va = list(csv.DictReader(open(args.val_csv)))
    kin_cols_full, vj_cols, meta_cols = split_features(rows_tr, args.vjepa_prefix)
    # Optional restriction to old_118 column set
    if args.old_118_cols:
        with open(args.old_118_cols) as f:
            old_118 = set(json.load(f))
        kin_cols = [c for c in kin_cols_full if c in old_118]
        print(f"restricted to old_118: {len(kin_cols)} (was {len(kin_cols_full)})")
    else:
        kin_cols = kin_cols_full
    print(f"kin cols: {len(kin_cols)}, vjepa cols: {len(vj_cols)}, meta cols: {len(meta_cols)}")

    y_tr = np.array([int(float(r["label"])) for r in rows_tr])
    y_va = np.array([int(float(r["label"])) for r in rows_va])

    # Split meta into "label-leak" (item35_*) and "non-leak" (dx_*, etc.)
    item35_cols = [c for c in meta_cols if c.startswith("item35")]
    nonleak_meta_cols = [c for c in meta_cols if not c.startswith("item35")]
    print(f"  meta breakdown: item35_*={len(item35_cols)}, non-leak={len(nonleak_meta_cols)}")

    # Subdivide V-JEPA features by suffix index for ablation
    # The augment script wrote 96 cols = PCA16 × (mean[16] + std[16] + |Δ|[16] + 3 FFT bands × 16 = 48)
    # First 48 = stats (mean, std, |Δ|), last 48 = FFT bands. Indices 0-47 vs 48-95.
    def parse_idx(name):
        try:
            return int(name.rsplit("_", 1)[-1])
        except Exception:
            return -1
    vj_with_idx = [(c, parse_idx(c)) for c in vj_cols]
    vj_fft_only = [c for c, i in vj_with_idx if i >= 48]  # FFT band features
    vj_stats_only = [c for c, i in vj_with_idx if i < 48]  # mean/std/|Δ|
    vj_top16 = [c for c, i in vj_with_idx if 0 <= i < 16]  # first PCA dim mean only
    print(f"  vjepa subsets: fft_only={len(vj_fft_only)}, stats_only={len(vj_stats_only)}, top16(mean only)={len(vj_top16)}")

    sets = {
        "kin_only": kin_cols,
        "kin_plus_dx": kin_cols + nonleak_meta_cols,
        "kin_plus_dx_plus_vjepa_all": kin_cols + nonleak_meta_cols + vj_cols,
        "kin_plus_dx_plus_vjepa_fftOnly": kin_cols + nonleak_meta_cols + vj_fft_only,
        "kin_plus_dx_plus_vjepa_statsOnly": kin_cols + nonleak_meta_cols + vj_stats_only,
        "kin_plus_dx_plus_vjepa_top16": kin_cols + nonleak_meta_cols + vj_top16,
        "kin_plus_all_meta": kin_cols + meta_cols,
        "kin_plus_all_meta_plus_vjepa": kin_cols + meta_cols + vj_cols,
        "vjepa_only": vj_cols,
    }
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Running {len(sets)} feature sets × {len(seeds)} seeds = {len(sets)*len(seeds)} fits")

    all_results = {}
    for set_name, cols in sets.items():
        if not cols:
            continue
        X_tr = to_matrix(rows_tr, cols)
        X_va = to_matrix(rows_va, cols)
        per_seed = []
        for s in seeds:
            res = fit_eval(X_tr, y_tr, X_va, y_va, s, args.n_estimators)
            per_seed.append({"seed": s, **res})
        qwks = [r["qwk"] for r in per_seed]
        accs = [r["acc"] for r in per_seed]
        all_results[set_name] = {
            "n_cols": len(cols),
            "n_seeds": len(seeds),
            "qwk_mean": float(np.mean(qwks)),
            "qwk_std": float(np.std(qwks)),
            "qwk_min": float(np.min(qwks)),
            "qwk_max": float(np.max(qwks)),
            "qwk_median": float(np.median(qwks)),
            "acc_mean": float(np.mean(accs)),
            "per_seed": per_seed,
        }
        print(f"\n{set_name:<30} cols={len(cols):>4}  qwk: mean={np.mean(qwks):.4f} std={np.std(qwks):.4f} "
              f"min={np.min(qwks):.4f} max={np.max(qwks):.4f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
