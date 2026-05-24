#!/usr/bin/env python
"""Add the OOF V-JEPA combiner probs (5 cols) to a kinematic CSV."""

from __future__ import annotations

import argparse
import csv
import numpy as np
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-npz", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--out-train-csv", required=True)
    p.add_argument("--out-val-csv", required=True)
    p.add_argument("--prefix", default="vjepaOOFprob")
    args = p.parse_args()

    d = np.load(args.oof_npz)
    train_probs = d["train_probs_cache_order"]
    val_probs = d["val_probs_cache_order"]
    train_idx = d["train_cache_sample_indices"]
    val_idx = d["val_cache_sample_indices"]
    K = train_probs.shape[1]

    fold_tr = list(csv.DictReader(open(args.fold_train_csv)))
    fold_va = list(csv.DictReader(open(args.fold_val_csv)))

    train_by_clip = {}
    for cache_idx, csv_row_idx in enumerate(train_idx):
        if 0 <= int(csv_row_idx) < len(fold_tr):
            clip = fold_tr[int(csv_row_idx)]["clip_path"]
            train_by_clip[clip] = train_probs[cache_idx]
    val_by_clip = {}
    for cache_idx, csv_row_idx in enumerate(val_idx):
        if 0 <= int(csv_row_idx) < len(fold_va):
            clip = fold_va[int(csv_row_idx)]["clip_path"]
            val_by_clip[clip] = val_probs[cache_idx]

    feat_names = [f"{args.prefix}_{k}" for k in range(K)]

    def write_aug(in_path, out_path, by_clip):
        rows = list(csv.DictReader(open(in_path)))
        out_fields = list(rows[0].keys()) + feat_names
        kept = 0
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            for r in rows:
                clip = r["clip_path"]
                vec = by_clip.get(clip)
                if vec is not None:
                    for n, v in zip(feat_names, vec):
                        r[n] = float(v)
                    kept += 1
                else:
                    for n in feat_names:
                        r[n] = 0.0
                w.writerow(r)
        print(f"wrote {out_path}: {kept}/{len(rows)} with OOF probs")

    write_aug(args.kin_train_csv, args.out_train_csv, train_by_clip)
    write_aug(args.kin_val_csv, args.out_val_csv, val_by_clip)


if __name__ == "__main__":
    main()
