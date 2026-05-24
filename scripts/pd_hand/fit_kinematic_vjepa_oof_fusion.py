#!/usr/bin/env python3
"""Fit non-leaky kinematic + V-JEPA score fusion from subject-level OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pd_hand import temporal_oof_stacker as tos
from scripts.pd_hand import train_finger_tapping_kinematic_baseline as kb


DEFAULT_BASE = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks"
)
DEFAULT_KIN_DIR = DEFAULT_BASE / "kinematic_baselines/item_3_4/fold_0"
DEFAULT_VJEPA_PROBS = (
    DEFAULT_BASE / "vjepa2_temporal_encoder/item_3_4/"
    "temporal_oof_stacker_top4_e35_subject_unmaskedfix_probs.probs.npz"
)
DEFAULT_OUT = (
    DEFAULT_BASE / "vjepa2_temporal_encoder/item_3_4/"
    "kinematic_vjepa_subject_oof_fusion_top4.json"
)
NUM_CLASSES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_BASE / "splits/item_3_4/fold_0_train.csv")
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_BASE / "splits/item_3_4/fold_0_val.csv")
    parser.add_argument("--train-features", type=Path, default=DEFAULT_KIN_DIR / "train_features_stride2.csv")
    parser.add_argument("--val-features", type=Path, default=DEFAULT_KIN_DIR / "val_features_stride2.csv")
    parser.add_argument("--vjepa-probs", type=Path, default=DEFAULT_VJEPA_PROBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--max-threshold-candidates", type=int, default=9)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vjepa = np.load(args.vjepa_probs, allow_pickle=True)
    vjepa_oof = vjepa["oof_probs"].astype(np.float32)
    vjepa_val = vjepa["val_probs"].astype(np.float32)
    y_train = vjepa["labels_train"].astype(np.int64)
    y_val = vjepa["labels_val"].astype(np.int64)

    train_rows = _features_in_cache_order(args.train_features, args.train_csv, y_train)
    val_rows = _features_in_cache_order(args.val_features, args.val_csv, y_val)
    groups = np.asarray([row["subject_id"] for row in train_rows], dtype=object)

    feature_sets = _feature_sets(train_rows)
    model_specs = _model_specs()
    kin_items = []
    for spec_name, feature_set_name, model_template in model_specs:
        feature_names = feature_sets[feature_set_name]
        x_train = _matrix(train_rows, feature_names)
        x_val = _matrix(val_rows, feature_names)
        kin_oof, kin_val = _fit_oof_and_full_val(
            model_template=model_template,
            x_train=x_train,
            y_train=y_train,
            groups=groups,
            x_val=x_val,
            inner_splits=args.inner_splits,
        )
        oof_pred = kin_oof.argmax(axis=1).astype(np.int64)
        val_pred = kin_val.argmax(axis=1).astype(np.int64)
        kin_item = {
            "name": spec_name,
            "feature_set": feature_set_name,
            "feature_count": len(feature_names),
            "oof_probs": kin_oof,
            "val_probs": kin_val,
            "oof_metrics": tos._metrics(y_train, oof_pred),
            "val_metrics": tos._metrics(y_val, val_pred),
        }
        kin_items.append(kin_item)
        print(
            f"kin {spec_name:34s} "
            f"oof={100*kin_item['oof_metrics']['acc']:.1f}/{kin_item['oof_metrics']['qwk']:.3f}/"
            f"{kin_item['oof_metrics']['mae']:.3f} "
            f"val={100*kin_item['val_metrics']['acc']:.1f}/{kin_item['val_metrics']['qwk']:.3f}/"
            f"{kin_item['val_metrics']['mae']:.3f}",
            flush=True,
        )

    vjepa_items = _vjepa_items(vjepa_oof, vjepa_val, y_train, y_val)
    rows = []
    rows.extend(_single_rows("kin", kin_items, y_train, y_val, args.max_threshold_candidates))
    rows.extend(_single_rows("vjepa", vjepa_items, y_train, y_val, args.max_threshold_candidates))
    for kin in kin_items:
        for vision in vjepa_items:
            rows.extend(_fusion_rows(
                kin=kin,
                vision=vision,
                y_train=y_train,
                y_val=y_val,
                max_threshold_candidates=args.max_threshold_candidates,
                weight_step=args.weight_step,
            ))

    rows_sorted_by_oof = sorted(rows, key=lambda r: (r["oof_qwk"], r["oof_acc"], -r["oof_mae"]), reverse=True)
    rows_sorted_by_val = sorted(rows, key=lambda r: (r["val_qwk"], r["val_acc"], -r["val_mae"]), reverse=True)

    print("\nTop by OOF:")
    for row in rows_sorted_by_oof[:12]:
        print(_format_row(row), flush=True)
    print("\nTop diagnostic by val:")
    for row in rows_sorted_by_val[:8]:
        print(_format_row(row), flush=True)

    output = {
        "note": (
            "Kinematic + V-JEPA fusion fit on fold-0 train subject-level OOF predictions. "
            "Held-out fold-0 validation labels are used only for final evaluation and diagnostics."
        ),
        "train_n": int(y_train.size),
        "val_n": int(y_val.size),
        "inner_splits": int(args.inner_splits),
        "vjepa_probs": str(args.vjepa_probs),
        "kinematic_models": [_strip_arrays(item) for item in kin_items],
        "vjepa_models": [_strip_arrays(item) for item in vjepa_items],
        "best_by_oof": rows_sorted_by_oof[0],
        "best_by_val_diagnostic": rows_sorted_by_val[0],
        "top_by_oof": rows_sorted_by_oof[: args.top_n],
        "top_by_val_diagnostic": rows_sorted_by_val[: args.top_n],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)


def _features_in_cache_order(feature_csv: Path, split_csv: Path, labels: np.ndarray) -> list[dict[str, str]]:
    with feature_csv.open(newline="") as handle:
        feature_by_row_id = {row["row_id"]: row for row in csv.DictReader(handle)}
    with split_csv.open(newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    order = _distributed_sampler_order(len(split_rows))
    rows = []
    for sample_idx in order:
        row_id = f"{split_csv.stem}:{sample_idx}"
        if row_id not in feature_by_row_id:
            raise SystemExit(f"Missing kinematic feature row_id={row_id} in {feature_csv}")
        rows.append(feature_by_row_id[row_id])
    feature_labels = np.asarray([int(float(row["label"])) for row in rows], dtype=np.int64)
    if not np.array_equal(feature_labels, labels):
        raise SystemExit(f"{feature_csv} labels do not match V-JEPA cache order")
    return rows


def _distributed_sampler_order(n_rows: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.randperm(n_rows, generator=generator).tolist()


def _feature_sets(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    ignore = {"row_id", "clip_path", "subject_id", "visit_id", "side", "dx", "label", "extract_error"}
    all_features = sorted(key for key in rows[0] if key not in ignore)
    old_features = kb._old_feature_names(all_features)
    feature_sets = {
        "all": all_features,
        "old_118": old_features,
        "meta": [
            key for key in all_features
            if key.startswith("dx_")
            or key.startswith("item35")
            or key in {"side_right", "duration_s_manifest", "start_s_manifest"}
        ],
        "old_openmeta": [
            key for key in old_features
            if key.startswith(("open_", "open_px_", "dx_", "item35"))
            or key in {"side_right", "duration_s_manifest"}
        ],
    }
    return {name: features for name, features in feature_sets.items() if features}


def _model_specs():
    return [
        (
            "old_118:extra_trees_old_seed98",
            "old_118",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=80,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight=None,
                    random_state=98,
                    n_jobs=2,
                ),
            ),
        ),
        (
            "old_118:extra_trees_old_seed0",
            "old_118",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=80,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=2,
                ),
            ),
        ),
        (
            "meta:extra_trees_old_seed0",
            "meta",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=80,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=2,
                ),
            ),
        ),
        (
            "old_openmeta:random_forest",
            "old_openmeta",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=1000,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=23,
                    n_jobs=2,
                ),
            ),
        ),
        (
            "meta:gradient_boosting",
            "meta",
            make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(random_state=29),
            ),
        ),
    ]


def _matrix(rows: list[dict[str, str]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[kb._float(row.get(key)) for key in feature_names] for row in rows], dtype=np.float64)


def _fit_oof_and_full_val(
    model_template: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_val: np.ndarray,
    inner_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((y_train.shape[0], NUM_CLASSES), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=4321)
    for fit_idx, hold_idx in splitter.split(x_train, y_train, groups):
        model = clone(model_template)
        model.fit(x_train[fit_idx], y_train[fit_idx])
        oof[hold_idx] = _predict_proba_full(model, x_train[hold_idx])
    full_model = clone(model_template)
    full_model.fit(x_train, y_train)
    val_probs = _predict_proba_full(full_model, x_val)
    return oof, val_probs


def _predict_proba_full(model: Any, x: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        pred = model.predict(x).astype(np.int64)
        probs = np.zeros((pred.shape[0], NUM_CLASSES), dtype=np.float32)
        probs[np.arange(pred.shape[0]), pred] = 1.0
        return probs
    raw = model.predict_proba(x)
    probs = np.zeros((x.shape[0], NUM_CLASSES), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is None:
        classes = np.arange(raw.shape[1])
    for col_idx, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int < NUM_CLASSES:
            probs[:, cls_int] = raw[:, col_idx]
    denom = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, np.maximum(denom, 1e-8), out=np.zeros_like(probs), where=denom > 0)


def _vjepa_items(vjepa_oof: np.ndarray, vjepa_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> list[dict[str, Any]]:
    items = []
    for idx in range(vjepa_oof.shape[0]):
        items.append(_prob_item(f"candidate_{idx}", vjepa_oof[idx], vjepa_val[idx], y_train, y_val))
    aggregates = {
        "top3_avg": [0, 1, 2],
        "top4_avg": [0, 1, 2, 3],
        "subject_oof_greedy_k3": [3, 2, 0],
    }
    for name, indices in aggregates.items():
        items.append(_prob_item(name, vjepa_oof[indices].mean(axis=0), vjepa_val[indices].mean(axis=0), y_train, y_val))
    return items


def _prob_item(name: str, oof_probs: np.ndarray, val_probs: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "oof_probs": oof_probs.astype(np.float32),
        "val_probs": val_probs.astype(np.float32),
        "oof_metrics": tos._metrics(y_train, oof_probs.argmax(axis=1).astype(np.int64)),
        "val_metrics": tos._metrics(y_val, val_probs.argmax(axis=1).astype(np.int64)),
    }


def _single_rows(prefix: str, items: list[dict[str, Any]], y_train: np.ndarray, y_val: np.ndarray, max_threshold_candidates: int):
    rows = []
    for item in items:
        oof_score = tos._expected_score(item["oof_probs"])
        val_score = tos._expected_score(item["val_probs"])
        row = _eval_score(
            name=f"{prefix}:{item['name']}",
            kind=f"{prefix}_score_threshold",
            oof_score=oof_score,
            val_score=val_score,
            y_train=y_train,
            y_val=y_val,
            max_threshold_candidates=max_threshold_candidates,
        )
        row["components"] = [{"type": prefix, "name": item["name"], "weight": 1.0}]
        rows.append(row)
    return rows


def _fusion_rows(
    kin: dict[str, Any],
    vision: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    max_threshold_candidates: int,
    weight_step: float,
):
    rows = []
    kin_oof_score = tos._expected_score(kin["oof_probs"])
    kin_val_score = tos._expected_score(kin["val_probs"])
    vision_oof_score = tos._expected_score(vision["oof_probs"])
    vision_val_score = tos._expected_score(vision["val_probs"])
    for weight in np.arange(0.0, 1.0 + 0.5 * weight_step, weight_step):
        oof_score = weight * kin_oof_score + (1.0 - weight) * vision_oof_score
        val_score = weight * kin_val_score + (1.0 - weight) * vision_val_score
        row = _eval_score(
            name=f"{kin['name']} + {vision['name']}",
            kind="kin_vjepa_score_threshold",
            oof_score=oof_score,
            val_score=val_score,
            y_train=y_train,
            y_val=y_val,
            max_threshold_candidates=max_threshold_candidates,
        )
        row["kin_weight"] = float(weight)
        row["vjepa_weight"] = float(1.0 - weight)
        row["components"] = [
            {"type": "kinematic", "name": kin["name"], "weight": float(weight)},
            {"type": "vjepa", "name": vision["name"], "weight": float(1.0 - weight)},
        ]
        rows.append(row)
    return rows


def _eval_score(
    name: str,
    kind: str,
    oof_score: np.ndarray,
    val_score: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    max_threshold_candidates: int,
) -> dict[str, Any]:
    thresholds, oof_pred = tos._fit_thresholds(oof_score, y_train, max_threshold_candidates)
    val_pred = tos._threshold_predict(val_score, thresholds)
    row = _row(name, kind, y_train, oof_pred, y_val, val_pred)
    row["thresholds"] = [float(x) for x in thresholds]
    return row


def _row(name: str, kind: str, y_train: np.ndarray, oof_pred: np.ndarray, y_val: np.ndarray, val_pred: np.ndarray) -> dict[str, Any]:
    oof = tos._metrics(y_train, oof_pred)
    val = tos._metrics(y_val, val_pred)
    return {
        "name": name,
        "kind": kind,
        "oof_acc": oof["acc"],
        "oof_qwk": oof["qwk"],
        "oof_mae": oof["mae"],
        "oof_counts": oof["counts"],
        "val_acc": val["acc"],
        "val_qwk": val["qwk"],
        "val_mae": val["mae"],
        "val_counts": val["counts"],
        "val_confusion": val["confusion"],
    }


def _strip_arrays(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in {"oof_probs", "val_probs"}}


def _format_row(row: dict[str, Any]) -> str:
    extra = ""
    if "kin_weight" in row:
        extra = f" kin_w={row['kin_weight']:.3f} vjepa_w={row['vjepa_weight']:.3f}"
    return (
        f"{row['kind']:28s} {row['name'][:58]:58s} "
        f"oof={100*row['oof_acc']:.1f}/{row['oof_qwk']:.3f}/{row['oof_mae']:.3f} "
        f"val={100*row['val_acc']:.1f}/{row['val_qwk']:.3f}/{row['val_mae']:.3f}{extra}"
    )


if __name__ == "__main__":
    main()
