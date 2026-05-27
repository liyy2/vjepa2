#!/usr/bin/env python
"""Join V-JEPA OOF probability features onto kinematic feature CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-npz", required=True)
    parser.add_argument("--fold-train-csv", required=True)
    parser.add_argument("--fold-val-csv", required=True)
    parser.add_argument("--kin-train-csv", required=True)
    parser.add_argument("--kin-val-csv", required=True)
    parser.add_argument("--out-train-csv", required=True)
    parser.add_argument("--out-val-csv", required=True)
    parser.add_argument("--prefix", default="vjepaOOFprob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof = np.load(args.oof_npz)
    train_probs = oof["train_probs_cache_order"]
    val_probs = oof["val_probs_cache_order"]
    train_index = oof["train_cache_sample_indices"]
    val_index = oof["val_cache_sample_indices"]

    train_fold_rows = read_csv(args.fold_train_csv)
    val_fold_rows = read_csv(args.fold_val_csv)
    train_by_clip = map_probs_to_clip(train_probs, train_index, train_fold_rows, "train")
    val_by_clip = map_probs_to_clip(val_probs, val_index, val_fold_rows, "val")

    feature_names = [f"{args.prefix}_{idx}" for idx in range(train_probs.shape[1])]
    write_augmented(args.kin_train_csv, args.out_train_csv, train_by_clip, feature_names)
    write_augmented(args.kin_val_csv, args.out_val_csv, val_by_clip, feature_names)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def map_probs_to_clip(
    probs: np.ndarray,
    sample_indices: np.ndarray,
    fold_rows: list[dict[str, str]],
    split_name: str,
) -> dict[str, np.ndarray]:
    if probs.shape[0] != sample_indices.shape[0]:
        raise ValueError(f"{split_name}: probs rows {probs.shape[0]} != index rows {sample_indices.shape[0]}")

    out: dict[str, np.ndarray] = {}
    for cache_idx, csv_idx in enumerate(sample_indices):
        row_idx = int(csv_idx)
        if not 0 <= row_idx < len(fold_rows):
            raise ValueError(f"{split_name}: cache row {cache_idx} has invalid sample_index={row_idx}")
        clip = fold_rows[row_idx]["clip_path"]
        if clip in out:
            raise ValueError(f"{split_name}: duplicate clip_path in OOF map: {clip}")
        out[clip] = probs[cache_idx]
    return out


def write_augmented(
    input_csv: str | Path,
    output_csv: str | Path,
    by_clip: dict[str, np.ndarray],
    feature_names: list[str],
) -> None:
    rows = read_csv(input_csv)
    seen = set()
    for row in rows:
        clip = row["clip_path"]
        if clip in seen:
            raise ValueError(f"{input_csv}: duplicate clip_path in kinematic CSV: {clip}")
        seen.add(clip)

    missing = [row["clip_path"] for row in rows if row["clip_path"] not in by_clip]
    if missing:
        raise ValueError(f"{input_csv}: missing OOF probabilities for {len(missing)} clips, first={missing[0]}")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + feature_names
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for name, value in zip(feature_names, by_clip[row["clip_path"]]):
                row[name] = float(value)
            writer.writerow(row)
    print(f"wrote {output_path}: {len(rows)}/{len(rows)} rows with OOF probabilities")


if __name__ == "__main__":
    main()
