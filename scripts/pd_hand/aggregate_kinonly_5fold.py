#!/usr/bin/env python
"""Aggregate the May 24 fair-baseline 5-fold multiseed scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DEFAULT_ROOT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4"
)
DEFAULT_OUT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19/r2_kin_scan/kinonly_5fold_aggregate.json"
)

COMPARISONS = [
    ("kin_only", "kin_only_plus_vjepa_all", "fair_kin_only_plus_vjepa_oof"),
    ("kin_plus_dx", "kin_plus_dx_plus_vjepa_all", "kin_plus_dx_reference"),
    ("kin_only", "vjepa_only", "vjepa_only_reference"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-json", default=DEFAULT_OUT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--file-name", default="multiseed_oof_probs_kinonly.json")
    parser.add_argument(
        "--fold0-file-name",
        default="multiseed_oof_probs_kinonly_clean.json",
        help="Historical fold-0 output name. Set empty to use --file-name for all folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    folds = [int(fold) for fold in args.folds.split(",") if fold.strip()]
    fold_data = {fold: load_fold(root, fold, args.file_name, args.fold0_file_name) for fold in folds}

    payload = {"root": str(root), "folds_loaded": folds, "comparisons": []}
    for base_key, hybrid_key, label in COMPARISONS:
        if not all(has_result(fold_data[fold], base_key) and has_result(fold_data[fold], hybrid_key) for fold in folds):
            print(f"skip {label}: missing {base_key} or {hybrid_key}")
            continue
        summary = summarize_comparison(fold_data, folds, base_key, hybrid_key, label)
        payload["comparisons"].append(summary)
        print_summary(summary)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


def load_fold(root: Path, fold: int, file_name: str, fold0_file_name: str) -> dict:
    names = [fold0_file_name, file_name] if fold == 0 and fold0_file_name else [file_name]
    tried = []
    for name in names:
        path = root / f"fold_{fold}" / "oof_stack" / name
        tried.append(str(path))
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("results", data)
    raise FileNotFoundError(" or ".join(tried))


def has_result(data: dict, key: str) -> bool:
    return key in data


def per_seed(data: dict, key: str) -> list[dict]:
    rows = data[key]["per_seed"]
    if not rows:
        raise ValueError(f"No per_seed rows for {key}")
    return rows


def paired_qwk(data: dict, base_key: str, hybrid_key: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    base_rows = per_seed(data, base_key)
    hybrid_rows = per_seed(data, hybrid_key)
    base_seeds = [int(row["seed"]) for row in base_rows]
    hybrid_seeds = [int(row["seed"]) for row in hybrid_rows]
    if base_seeds != hybrid_seeds:
        raise ValueError(f"Seed mismatch: {base_key} {base_seeds} vs {hybrid_key} {hybrid_seeds}")
    base = np.asarray([row["qwk"] for row in base_rows], dtype=np.float64)
    hybrid = np.asarray([row["qwk"] for row in hybrid_rows], dtype=np.float64)
    return base, hybrid, base_seeds


def summarize_comparison(
    fold_data: dict[int, dict],
    folds: list[int],
    base_key: str,
    hybrid_key: str,
    label: str,
) -> dict:
    per_fold = {}
    all_base = []
    all_hybrid = []
    for fold in folds:
        base, hybrid, seeds = paired_qwk(fold_data[fold], base_key, hybrid_key)
        delta = hybrid - base
        per_fold[str(fold)] = {
            "seeds": seeds,
            "base_qwk": base.tolist(),
            "hybrid_qwk": hybrid.tolist(),
            "base_mean": float(base.mean()),
            "base_std": float(base.std()),
            "hybrid_mean": float(hybrid.mean()),
            "hybrid_std": float(hybrid.std()),
            "delta_mean": float(delta.mean()),
            "wins": int((delta > 0).sum()),
            "n_seeds": int(len(seeds)),
        }
        all_base.append(base)
        all_hybrid.append(hybrid)

    base_means = np.asarray([per_fold[str(fold)]["base_mean"] for fold in folds])
    hybrid_means = np.asarray([per_fold[str(fold)]["hybrid_mean"] for fold in folds])
    pooled_base = np.concatenate(all_base)
    pooled_hybrid = np.concatenate(all_hybrid)
    pooled_delta = pooled_hybrid - pooled_base
    fold_delta = hybrid_means - base_means

    return {
        "label": label,
        "base_key": base_key,
        "hybrid_key": hybrid_key,
        "per_fold": per_fold,
        "base_across_fold_mean": float(base_means.mean()),
        "base_across_fold_std": float(base_means.std()),
        "hybrid_across_fold_mean": float(hybrid_means.mean()),
        "hybrid_across_fold_std": float(hybrid_means.std()),
        "delta_across_fold_mean": float(fold_delta.mean()),
        "positive_folds": int((fold_delta > 0).sum()),
        "n_folds": int(len(folds)),
        "delta_pooled_mean": float(pooled_delta.mean()),
        "delta_pooled_median": float(np.median(pooled_delta)),
        "wins_pooled": int((pooled_delta > 0).sum()),
        "n_pooled": int(len(pooled_delta)),
        "wilcoxon_pooled_one_sided": wilcoxon_pvalue(pooled_hybrid, pooled_base),
        "wilcoxon_fold_mean_one_sided": wilcoxon_pvalue(hybrid_means, base_means),
    }


def wilcoxon_pvalue(hybrid: np.ndarray, base: np.ndarray) -> float:
    try:
        return float(wilcoxon(hybrid, base, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['label']} ===")
    for fold, row in summary["per_fold"].items():
        print(
            f"fold {fold}: base {row['base_mean']:+.4f}+/-{row['base_std']:.4f}  "
            f"hybrid {row['hybrid_mean']:+.4f}+/-{row['hybrid_std']:.4f}  "
            f"delta {row['delta_mean']:+.4f}  wins {row['wins']}/{row['n_seeds']}"
        )
    print(
        f"mean: base {summary['base_across_fold_mean']:+.4f}  "
        f"hybrid {summary['hybrid_across_fold_mean']:+.4f}  "
        f"delta {summary['delta_across_fold_mean']:+.4f}  "
        f"positive folds {summary['positive_folds']}/{summary['n_folds']}"
    )
    print(
        f"pooled seed-fold: wins {summary['wins_pooled']}/{summary['n_pooled']}  "
        f"p={summary['wilcoxon_pooled_one_sided']:.4g}; "
        f"fold-mean p={summary['wilcoxon_fold_mean_one_sided']:.4g}"
    )


if __name__ == "__main__":
    main()
