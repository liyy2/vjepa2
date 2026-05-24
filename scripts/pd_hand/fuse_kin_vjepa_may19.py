#!/usr/bin/env python
"""Late-fusion of kinematic + V-JEPA predictions on may19 fold 0.

Aligns kinematic predictions (in lex-sorted CSV row order) with V-JEPA cache
predictions (in shuffled cache order, sample_index in coverage). Searches the
fusion weight α ∈ [0, 1] that maximizes held-out QWK. NOTE: searching α on
the same val set that we evaluate is val-tuned — report this rather than the
honest fusion number. Useful as an upper-bound diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kin-json", required=True,
                   help="results_stride2.json from train_finger_tapping_kinematic_baseline.py")
    p.add_argument("--vjepa-json", required=True,
                   help="result.json from train_corn_combiner.py (or train_temporal_encoder.py)")
    p.add_argument("--vjepa-cache", required=True,
                   help="V-JEPA cache .npz (val split) to read sample_index from coverage")
    p.add_argument("--fold-val-csv", required=True,
                   help="fold_0_val.csv to map sample_index to CSV row")
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--alpha-step", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    # 1. Kinematic preds + labels (lex-sorted by row_id)
    kin_data = json.loads(Path(args.kin_json).read_text())["best"]
    kin_preds = np.array(kin_data["predictions"], dtype=np.int64)
    kin_labels = np.array(kin_data["labels"], dtype=np.int64)

    # The fold_0_val.csv row indices, in lex-sorted row_id order
    fold_rows = list(csv.DictReader(open(args.fold_val_csv)))
    n = len(fold_rows)
    lex_order = sorted(range(n), key=lambda i: f"fold_0_val:{i}")  # same key used in kinematic baseline
    # csv_row_idx_for_kin_position[k] = csv row index of the k-th kin prediction
    csv_row_idx_for_kin = lex_order

    # 2. V-JEPA preds + val_probs from combiner OR from tstats
    vj_data = json.loads(Path(args.vjepa_json).read_text())
    # combiner schema: best.metrics.preds + best.metrics.val_probs
    best = vj_data.get("best", vj_data.get("best_single_model", {}))
    metrics = best.get("metrics", best)
    vj_preds = np.array(metrics.get("preds") or metrics.get("predictions") or [], dtype=np.int64)
    vj_probs = metrics.get("val_probs")
    if vj_probs is None:
        # convert to one-hot from preds
        vj_probs = np.zeros((len(vj_preds), args.num_classes), dtype=np.float64)
        for i, y in enumerate(vj_preds):
            vj_probs[i, y] = 1.0
    else:
        vj_probs = np.asarray(vj_probs, dtype=np.float64)
    vj_labels = np.array(metrics.get("labels") or best.get("labels") or [], dtype=np.int64)

    # 3. Determine V-JEPA cache row → CSV row index (via sample_index)
    cache = np.load(args.vjepa_cache)
    if "coverage" not in cache.files:
        raise SystemExit("V-JEPA cache has no coverage field, cannot align")
    coverage = [json.loads(str(s)) for s in cache["coverage"]]
    cache_csv_idx = [int(round(float(c.get("sample_index", -1)))) for c in coverage]
    # Map: V-JEPA position k → CSV row idx
    vj_csv_idx = cache_csv_idx

    # 4. Express both kin_probs and vj_probs in CSV-natural order
    # Kin: position k has csv_idx = csv_row_idx_for_kin[k]
    # VJ:  position k has csv_idx = vj_csv_idx[k]
    csv_kin = np.full((n, args.num_classes), np.nan, dtype=np.float64)
    csv_vj = np.full((n, args.num_classes), np.nan, dtype=np.float64)
    csv_labels = np.full(n, -1, dtype=np.int64)
    # kin one-hot
    for k_pos, kin_csv_idx in enumerate(csv_row_idx_for_kin):
        csv_kin[kin_csv_idx, :] = 0.0
        csv_kin[kin_csv_idx, kin_preds[k_pos]] = 1.0
        csv_labels[kin_csv_idx] = kin_labels[k_pos]
    # vj probs
    for k_pos, c_idx in enumerate(vj_csv_idx):
        if c_idx < 0 or c_idx >= n:
            continue
        csv_vj[c_idx, :] = vj_probs[k_pos]
    # Verify alignment of labels
    if (csv_labels < 0).any():
        print(f"warning: {int((csv_labels < 0).sum())} CSV rows have no kinematic label")
    # Drop unaligned rows
    valid = (~np.isnan(csv_kin).any(axis=1)) & (~np.isnan(csv_vj).any(axis=1)) & (csv_labels >= 0)
    print(f"aligned rows: {valid.sum()} of {n}")

    kin = csv_kin[valid]
    vj = csv_vj[valid]
    y = csv_labels[valid]

    print(f"\nkin alone: qwk={cohen_kappa_score(y, kin.argmax(1), weights='quadratic'):.4f} "
          f"acc={accuracy_score(y, kin.argmax(1)):.4f}")
    print(f"vj alone:  qwk={cohen_kappa_score(y, vj.argmax(1), weights='quadratic'):.4f} "
          f"acc={accuracy_score(y, vj.argmax(1)):.4f}")

    # Sweep fusion weight
    best_alpha, best_qwk, best_metrics = None, -2.0, None
    print(f"\n{'alpha':>6} {'qwk':>8} {'acc':>8} {'mae':>8}")
    for a in np.arange(0.0, 1.01, args.alpha_step):
        fused = a * kin + (1 - a) * vj
        yhat = fused.argmax(axis=1)
        qwk = cohen_kappa_score(y, yhat, weights="quadratic")
        acc = accuracy_score(y, yhat)
        mae = mean_absolute_error(y, yhat)
        print(f"{a:>6.2f} {qwk:>8.4f} {acc:>8.4f} {mae:>8.4f}")
        if qwk > best_qwk:
            best_alpha = float(a)
            best_qwk = float(qwk)
            best_metrics = {"alpha": best_alpha, "qwk": qwk, "acc": acc, "mae": mae}

    print(f"\nBEST: alpha={best_alpha}  qwk={best_qwk:.4f}  acc={best_metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
