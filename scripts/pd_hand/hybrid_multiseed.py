#!/usr/bin/env python
"""Evaluate fair kinematic baselines with optional V-JEPA OOF probabilities.

The May 24 result intentionally excludes diagnosis and item 3.5 features from
the fair `kin_only` baseline.  This script enforces that boundary in code and
fails if a leaked item 3.5 feature would enter any evaluated feature set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.pipeline import make_pipeline


ID_COLUMNS = {"row_id", "subject_id", "visit_id", "side", "clip_path", "label", "side_right_orig"}
CONTEXT_COLUMNS = ("side_right", "duration_s_manifest", "start_s_manifest")
LEGACY_INERT_COLUMNS = ("dx", "frame_count_manifest")
INERT_COLUMNS = set(LEGACY_INERT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--vjepa-prefix", default="vjepaOOFprob")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--n-estimators", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = read_csv(args.train_csv)
    val_rows = read_csv(args.val_csv)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    columns = split_feature_columns(train_rows, args.vjepa_prefix)
    y_train = labels(train_rows)
    y_val = labels(val_rows)

    feature_sets = {
        "kin_only": columns["kinematic"],
        "kin_only_plus_vjepa_all": columns["kinematic"] + columns["vjepa"],
        "kin_plus_dx": columns["kin_dx_reference"],
        "kin_plus_dx_plus_vjepa_all": columns["kin_dx_reference"] + columns["vjepa"],
        "vjepa_only": columns["vjepa"],
    }
    for name, feature_cols in feature_sets.items():
        assert_no_item35(name, feature_cols)

    payload = {
        "inputs": {
            "train_csv": str(args.train_csv),
            "val_csv": str(args.val_csv),
            "vjepa_prefix": args.vjepa_prefix,
            "seeds": seeds,
            "n_estimators": args.n_estimators,
        },
        "column_counts": {name: len(cols) for name, cols in columns.items()},
        "feature_sets": {name: cols for name, cols in feature_sets.items()},
        "results": {},
    }

    print(
        "columns: "
        f"kinematic={len(columns['kinematic'])}, context={len(columns['context'])}, "
        f"legacy_inert={len(columns['legacy_inert'])}, dx={len(columns['dx'])}, "
        f"kin_dx_reference={len(columns['kin_dx_reference'])}, "
        f"vjepa={len(columns['vjepa'])}, excluded={len(columns['excluded'])}"
    )
    for set_name, feature_cols in feature_sets.items():
        if not feature_cols:
            print(f"skip {set_name}: no columns")
            continue
        X_train = to_matrix(train_rows, feature_cols)
        X_val = to_matrix(val_rows, feature_cols)
        per_seed = [
            {"seed": seed, **fit_eval(X_train, y_train, X_val, y_val, seed, args.n_estimators)}
            for seed in seeds
        ]
        qwks = np.asarray([row["qwk"] for row in per_seed], dtype=np.float64)
        accs = np.asarray([row["acc"] for row in per_seed], dtype=np.float64)
        maes = np.asarray([row["mae"] for row in per_seed], dtype=np.float64)
        payload["results"][set_name] = {
            "n_cols": len(feature_cols),
            "n_seeds": len(seeds),
            "qwk_mean": float(qwks.mean()),
            "qwk_std": float(qwks.std()),
            "qwk_min": float(qwks.min()),
            "qwk_median": float(np.median(qwks)),
            "qwk_max": float(qwks.max()),
            "acc_mean": float(accs.mean()),
            "mae_mean": float(maes.mean()),
            "per_seed": per_seed,
        }
        print(
            f"{set_name:<30} cols={len(feature_cols):>4} "
            f"qwk={qwks.mean():+.4f} +/- {qwks.std():.4f} "
            f"acc={accs.mean():.4f} mae={maes.mean():.4f}"
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def split_feature_columns(rows: list[dict[str, str]], vjepa_prefix: str) -> dict[str, list[str]]:
    all_columns = list(rows[0])
    candidate_columns = [col for col in all_columns if col not in ID_COLUMNS]
    vjepa_cols = [col for col in all_columns if col.startswith(vjepa_prefix)]
    dx_cols = [col for col in all_columns if col.startswith("dx_")]
    legacy_meta = [
        col
        for col in candidate_columns
        if col.startswith("dx_") or col.startswith("item35") or col in CONTEXT_COLUMNS
    ]
    legacy_kin = [
        col
        for col in candidate_columns
        if col not in vjepa_cols and col not in legacy_meta
    ]
    legacy_nonleak_meta = [col for col in legacy_meta if not col.startswith("item35")]
    excluded: list[str] = []
    kinematic: list[str] = []

    for col in all_columns:
        if col in ID_COLUMNS or col in CONTEXT_COLUMNS or col in INERT_COLUMNS:
            excluded.append(col)
            continue
        if col.startswith("item35"):
            excluded.append(col)
            continue
        if col.startswith(vjepa_prefix) or col.startswith("dx_"):
            continue
        if has_finite_number(rows, col):
            kinematic.append(col)
        else:
            excluded.append(col)

    return {
        "kinematic": kinematic,
        "context": [col for col in CONTEXT_COLUMNS if col in all_columns and has_finite_number(rows, col)],
        "legacy_inert": [col for col in LEGACY_INERT_COLUMNS if col in all_columns],
        "dx": dx_cols,
        "kin_dx_reference": legacy_kin + legacy_nonleak_meta,
        "vjepa": vjepa_cols,
        "excluded": excluded,
    }


def has_finite_number(rows: Iterable[dict[str, str]], column: str) -> bool:
    for row in rows:
        try:
            if np.isfinite(float(row.get(column, "nan"))):
                return True
        except ValueError:
            continue
    return False


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(float(row["label"])) for row in rows], dtype=np.int64)


def to_matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    X = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            try:
                value = float(row.get(col, "nan"))
            except ValueError:
                continue
            if np.isfinite(value):
                X[i, j] = value
    return X


def fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_estimators: int,
) -> dict[str, float]:
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
    model.fit(X_train, y_train)
    pred = model.predict(X_val).astype(np.int64)
    return {
        "qwk": float(cohen_kappa_score(y_val, pred, weights="quadratic")),
        "acc": float(accuracy_score(y_val, pred)),
        "mae": float(mean_absolute_error(y_val, pred)),
    }


def assert_no_item35(set_name: str, columns: list[str]) -> None:
    leaked = [col for col in columns if col.startswith("item35")]
    if leaked:
        raise ValueError(f"{set_name} includes item 3.5 leakage columns: {leaked[:8]}")


if __name__ == "__main__":
    main()
