#!/usr/bin/env python
"""Train a classical finger-tapping baseline from MediaPipe hand kinematics.

This is intended as a fast sanity check against frozen V-JEPA probes for the PD
hand-task folds.  It extracts per-clip thumb/index opening features, joins
optional same-side item 3.5 labels as context, and evaluates several small
sklearn classifiers on the held-out fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_prominences, periodogram
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_BASE = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks"
)
DEFAULT_OUT = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0"
)


@dataclass(frozen=True)
class ClipRecord:
    row_id: str
    clip_path: str
    subject_id: str
    visit_id: str
    side: str
    dx: str
    label: int
    frame_count: float
    duration_s: float
    start_s: float
    item35_same: int | None = None
    item35_other: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=str(DEFAULT_BASE / "splits/item_3_4/fold_0_train.csv"))
    parser.add_argument("--val-csv", default=str(DEFAULT_BASE / "splits/item_3_4/fold_0_val.csv"))
    parser.add_argument("--manifest-csv", default=str(DEFAULT_BASE / "hand_clip_manifest.csv"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means use the whole clip after stride.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item35_lookup = _item35_lookup(args.manifest_csv)
    train_records = _read_records(args.train_csv, item35_lookup)
    val_records = _read_records(args.val_csv, item35_lookup)

    train_feature_csv = out_dir / f"train_features_stride{args.frame_stride}.csv"
    val_feature_csv = out_dir / f"val_features_stride{args.frame_stride}.csv"
    train_rows = _features_for_records(
        train_records,
        train_feature_csv,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        force=args.force,
    )
    val_rows = _features_for_records(
        val_records,
        val_feature_csv,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        force=args.force,
    )

    results = _fit_and_eval(train_rows, val_rows)
    result_path = out_dir / f"results_stride{args.frame_stride}.json"
    with result_path.open("w") as handle:
        json.dump(results, handle, indent=2)

    best = results["best"]
    print(
        f"best={best['model']} val_acc={best['val_acc']:.5f} "
        f"qwk={best['qwk']:.5f} mae={best['mae']:.5f}"
    )
    print("confusion_matrix=", best["confusion_matrix"])
    print(f"wrote {result_path}")


def _read_records(csv_path: str, item35_lookup: dict[tuple[str, str, str], int]) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    with open(csv_path, newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle)):
            side = row.get("side", "")
            other_side = "Left" if side == "Right" else "Right"
            key = (row["subject_id"], row["visit_id"], side)
            other_key = (row["subject_id"], row["visit_id"], other_side)
            records.append(
                ClipRecord(
                    row_id=f"{Path(csv_path).stem}:{idx}",
                    clip_path=row["clip_path"],
                    subject_id=row["subject_id"],
                    visit_id=row["visit_id"],
                    side=side,
                    dx=row.get("dx", ""),
                    label=int(float(row["label"])),
                    frame_count=_float(row.get("frame_count")),
                    duration_s=_float(row.get("duration_s")),
                    start_s=_float(row.get("start_s")),
                    item35_same=item35_lookup.get(key),
                    item35_other=item35_lookup.get(other_key),
                )
            )
    return records


def _item35_lookup(manifest_csv: str) -> dict[tuple[str, str, str], int]:
    lookup: dict[tuple[str, str, str], int] = {}
    with open(manifest_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("task_group") != "hand_movements" or not row.get("label"):
                continue
            lookup[(row["subject_id"], row["visit_id"], row["side"])] = int(float(row["label"]))
    return lookup


def _features_for_records(
    records: list[ClipRecord],
    cache_path: Path,
    frame_stride: int,
    max_frames: int,
    num_workers: int,
    force: bool,
) -> list[dict[str, Any]]:
    if cache_path.exists() and not force:
        with cache_path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = {
            executor.submit(_extract_one, record, frame_stride, max_frames): record
            for record in records
        }
        for i, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = _base_feature_row(record)
                row["extract_error"] = repr(exc)
            rows.append(row)
            if i % 25 == 0 or i == len(records):
                print(f"extracted {i}/{len(records)} -> {cache_path.name}", flush=True)

    rows.sort(key=lambda r: r["row_id"])
    fieldnames = sorted({key for row in rows for key in row})
    with cache_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _extract_one(record: ClipRecord, frame_stride: int, max_frames: int) -> dict[str, Any]:
    import mediapipe as mp

    row = _base_feature_row(record)
    cap = cv2.VideoCapture(record.clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    target_side = record.side.capitalize()
    flipped_side = "Left" if target_side == "Right" else "Right"
    streams: dict[str, dict[str, list[float]]] = {
        "": _empty_stream(),
        "flip_": _empty_stream(),
        "largest_": _empty_stream(),
    }
    detected = 0
    seen = 0
    used = 0
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue
            if max_frames > 0 and used >= max_frames:
                break
            used += 1
            seen += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            result = hands.process(frame_rgb)
            frame_index += 1
            if not result.multi_hand_landmarks:
                continue
            hand_idx = _select_hand_index(result, target_side, fallback=True)
            if hand_idx is None:
                continue
            detected += 1
            _append_hand_measurements(
                streams[""],
                result.multi_hand_landmarks[hand_idx],
                width=w,
                height=h,
            )
            flip_idx = _select_hand_index(result, flipped_side, fallback=False)
            if flip_idx is not None:
                _append_hand_measurements(
                    streams["flip_"],
                    result.multi_hand_landmarks[flip_idx],
                    width=w,
                    height=h,
                )
            largest_idx = _largest_hand_index(result, width=w, height=h)
            if largest_idx is not None:
                _append_hand_measurements(
                    streams["largest_"],
                    result.multi_hand_landmarks[largest_idx],
                    width=w,
                    height=h,
                )
    finally:
        hands.close()
        cap.release()

    sample_rate = fps / max(1, frame_stride)
    for prefix, stream in streams.items():
        row.update(_series_features(np.asarray(stream["open"], dtype=np.float32), f"{prefix}open", sample_rate))
        row.update(_series_features(np.asarray(stream["open_px"], dtype=np.float32), f"{prefix}open_px", sample_rate))
        row.update(_series_features(np.asarray(stream["wrist_x"], dtype=np.float32), f"{prefix}wrist_x", sample_rate))
        row.update(_series_features(np.asarray(stream["wrist_y"], dtype=np.float32), f"{prefix}wrist_y", sample_rate))
        row.update(_series_features(np.asarray(stream["bbox_area"], dtype=np.float32), f"{prefix}bbox_area", sample_rate))
    row["frames_seen"] = seen
    row["frames_with_hand"] = detected
    row["hand_detect_rate"] = detected / max(seen, 1)
    return row


def _empty_stream() -> dict[str, list[float]]:
    return {
        "open": [],
        "open_px": [],
        "wrist_x": [],
        "wrist_y": [],
        "bbox_area": [],
    }


def _append_hand_measurements(stream: dict[str, list[float]], landmarks: Any, width: int, height: int) -> None:
    points = np.array(
        [(lm.x * width, lm.y * height, lm.z * width) for lm in landmarks.landmark],
        dtype=np.float32,
    )
    thumb_tip = points[4, :2]
    index_tip = points[8, :2]
    wrist = points[0, :2]
    middle_mcp = points[9, :2]
    scale = max(float(np.linalg.norm(wrist - middle_mcp)), 1.0)
    raw_opening = float(np.linalg.norm(thumb_tip - index_tip))
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)

    stream["open"].append(raw_opening / scale)
    stream["open_px"].append(raw_opening)
    stream["wrist_x"].append(float(wrist[0] / max(width, 1)))
    stream["wrist_y"].append(float(wrist[1] / max(height, 1)))
    stream["bbox_area"].append(float(np.prod(max_xy - min_xy) / max(width * height, 1)))


def _largest_hand_index(result: Any, width: int, height: int) -> int | None:
    if not result.multi_hand_landmarks:
        return None
    best_idx: int | None = None
    best_area = -1.0
    for idx, landmarks in enumerate(result.multi_hand_landmarks):
        points = np.array([(lm.x * width, lm.y * height) for lm in landmarks.landmark], dtype=np.float32)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        area = float(np.prod(max_xy - min_xy))
        if area > best_area:
            best_area = area
            best_idx = idx
    return best_idx


def _select_hand_index(result: Any, target_side: str, fallback: bool) -> int | None:
    if not result.multi_hand_landmarks:
        return None
    handedness = result.multi_handedness or []
    for idx, hand_info in enumerate(handedness):
        if idx >= len(result.multi_hand_landmarks):
            continue
        label = hand_info.classification[0].label if hand_info.classification else ""
        if label == target_side:
            return idx
    return 0 if fallback else None


def _series_features(values: np.ndarray, prefix: str, sample_rate: float) -> dict[str, float]:
    out: dict[str, float] = {}
    if values.size == 0:
        for name in (
            "n",
            "mean",
            "std",
            "min",
            "max",
            "range",
            "p10",
            "p50",
            "p90",
            "slope",
            "first_mean",
            "last_mean",
            "decrement",
            "velocity_mean_abs",
            "velocity_max_abs",
            "peak_count",
            "peak_rate_hz",
            "peak_interval_mean",
            "peak_interval_std",
            "peak_interval_cv",
            "peak_value_mean",
            "peak_value_std",
            "peak_prominence_mean",
            "peak_prominence_std",
            "trough_count",
            "trough_rate_hz",
            "trough_value_mean",
            "cycle_amplitude_mean",
            "cycle_amplitude_std",
            "dominant_freq_hz",
            "dominant_power",
            "bandpower_0p5_2_hz",
            "bandpower_2_5_hz",
            "bandpower_5_8_hz",
            "velocity_std",
            "accel_mean_abs",
            "accel_std",
            "zero_cross_rate",
            "iqr",
            "mad",
        ):
            out[f"{prefix}_{name}"] = np.nan
        return out
    values = values.astype(np.float64)
    out[f"{prefix}_n"] = float(values.size)
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_range"] = float(np.max(values) - np.min(values))
    out[f"{prefix}_p10"] = float(np.percentile(values, 10))
    out[f"{prefix}_p50"] = float(np.percentile(values, 50))
    out[f"{prefix}_p90"] = float(np.percentile(values, 90))
    out[f"{prefix}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    out[f"{prefix}_mad"] = float(np.median(np.abs(values - np.median(values))))
    x = np.arange(values.size, dtype=np.float64)
    out[f"{prefix}_slope"] = float(np.polyfit(x, values, 1)[0]) if values.size >= 2 else 0.0
    thirds = np.array_split(values, 3)
    first_mean = float(np.mean(thirds[0]))
    last_mean = float(np.mean(thirds[-1]))
    out[f"{prefix}_first_mean"] = first_mean
    out[f"{prefix}_last_mean"] = last_mean
    out[f"{prefix}_decrement"] = float((first_mean - last_mean) / (abs(first_mean) + 1e-6))
    velocity = np.diff(values)
    out[f"{prefix}_velocity_mean_abs"] = float(np.mean(np.abs(velocity))) if velocity.size else 0.0
    out[f"{prefix}_velocity_max_abs"] = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    out[f"{prefix}_velocity_std"] = float(np.std(velocity)) if velocity.size else 0.0
    accel = np.diff(velocity)
    out[f"{prefix}_accel_mean_abs"] = float(np.mean(np.abs(accel))) if accel.size else 0.0
    out[f"{prefix}_accel_std"] = float(np.std(accel)) if accel.size else 0.0
    centered = values - np.nanmedian(values)
    out[f"{prefix}_zero_cross_rate"] = (
        float(np.mean(np.diff(np.signbit(centered)) != 0)) if values.size >= 2 else 0.0
    )
    prominence = max(float(np.nanstd(centered)) * 0.25, 1e-6)
    peaks, _ = find_peaks(centered, prominence=prominence, distance=max(1, int(sample_rate * 0.08)))
    troughs, _ = find_peaks(-centered, prominence=prominence, distance=max(1, int(sample_rate * 0.08)))
    duration = values.size / max(sample_rate, 1e-6)
    out[f"{prefix}_peak_count"] = float(peaks.size)
    out[f"{prefix}_peak_rate_hz"] = float(peaks.size / max(duration, 1e-6))
    out[f"{prefix}_trough_count"] = float(troughs.size)
    out[f"{prefix}_trough_rate_hz"] = float(troughs.size / max(duration, 1e-6))
    if peaks.size:
        peak_values = values[peaks]
        out[f"{prefix}_peak_value_mean"] = float(np.mean(peak_values))
        out[f"{prefix}_peak_value_std"] = float(np.std(peak_values))
        prominences = peak_prominences(centered, peaks)[0]
        out[f"{prefix}_peak_prominence_mean"] = float(np.mean(prominences))
        out[f"{prefix}_peak_prominence_std"] = float(np.std(prominences))
    else:
        out[f"{prefix}_peak_value_mean"] = np.nan
        out[f"{prefix}_peak_value_std"] = np.nan
        out[f"{prefix}_peak_prominence_mean"] = np.nan
        out[f"{prefix}_peak_prominence_std"] = np.nan
    if peaks.size >= 2:
        intervals = np.diff(peaks) / max(sample_rate, 1e-6)
        interval_mean = float(np.mean(intervals))
        out[f"{prefix}_peak_interval_mean"] = interval_mean
        out[f"{prefix}_peak_interval_std"] = float(np.std(intervals))
        out[f"{prefix}_peak_interval_cv"] = float(np.std(intervals) / (abs(interval_mean) + 1e-6))
    else:
        out[f"{prefix}_peak_interval_mean"] = np.nan
        out[f"{prefix}_peak_interval_std"] = np.nan
        out[f"{prefix}_peak_interval_cv"] = np.nan
    if troughs.size:
        trough_values = values[troughs]
        out[f"{prefix}_trough_value_mean"] = float(np.mean(trough_values))
    else:
        trough_values = np.asarray([], dtype=np.float64)
        out[f"{prefix}_trough_value_mean"] = np.nan
    if peaks.size and troughs.size:
        cycle_amplitudes = []
        for peak in peaks:
            nearest_trough = troughs[int(np.argmin(np.abs(troughs - peak)))]
            cycle_amplitudes.append(values[peak] - values[nearest_trough])
        out[f"{prefix}_cycle_amplitude_mean"] = float(np.mean(cycle_amplitudes))
        out[f"{prefix}_cycle_amplitude_std"] = float(np.std(cycle_amplitudes))
    else:
        out[f"{prefix}_cycle_amplitude_mean"] = np.nan
        out[f"{prefix}_cycle_amplitude_std"] = np.nan
    if values.size >= 8:
        freqs, power = periodogram(centered, fs=sample_rate)
        mask = (freqs >= 0.5) & (freqs <= 8.0)
        out[f"{prefix}_bandpower_0p5_2_hz"] = float(np.sum(power[(freqs >= 0.5) & (freqs < 2.0)]))
        out[f"{prefix}_bandpower_2_5_hz"] = float(np.sum(power[(freqs >= 2.0) & (freqs < 5.0)]))
        out[f"{prefix}_bandpower_5_8_hz"] = float(np.sum(power[(freqs >= 5.0) & (freqs <= 8.0)]))
        if np.any(mask):
            masked_freqs = freqs[mask]
            masked_power = power[mask]
            idx = int(np.argmax(masked_power))
            out[f"{prefix}_dominant_freq_hz"] = float(masked_freqs[idx])
            out[f"{prefix}_dominant_power"] = float(masked_power[idx])
        else:
            out[f"{prefix}_dominant_freq_hz"] = np.nan
            out[f"{prefix}_dominant_power"] = np.nan
    else:
        out[f"{prefix}_dominant_freq_hz"] = np.nan
        out[f"{prefix}_dominant_power"] = np.nan
        out[f"{prefix}_bandpower_0p5_2_hz"] = np.nan
        out[f"{prefix}_bandpower_2_5_hz"] = np.nan
        out[f"{prefix}_bandpower_5_8_hz"] = np.nan
    return out


def _base_feature_row(record: ClipRecord) -> dict[str, Any]:
    row = {
        "row_id": record.row_id,
        "clip_path": record.clip_path,
        "subject_id": record.subject_id,
        "visit_id": record.visit_id,
        "side": record.side,
        "dx": record.dx,
        "label": record.label,
        "frame_count_manifest": record.frame_count,
        "duration_s_manifest": record.duration_s,
        "start_s_manifest": record.start_s,
        "item35_same": -1 if record.item35_same is None else record.item35_same,
        "item35_other": -1 if record.item35_other is None else record.item35_other,
    }
    row["side_right"] = 1.0 if record.side == "Right" else 0.0
    for dx in ("HC", "NDC", "PD", "PPD"):
        row[f"dx_{dx}"] = 1.0 if record.dx == dx else 0.0
    for score in range(5):
        row[f"item35_same_{score}"] = 1.0 if record.item35_same == score else 0.0
        row[f"item35_other_{score}"] = 1.0 if record.item35_other == score else 0.0
    return row


def _fit_and_eval(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ignore = {"row_id", "clip_path", "subject_id", "visit_id", "side", "dx", "label", "extract_error"}
    y_train = np.asarray([int(float(row["label"])) for row in train_rows], dtype=np.int64)
    y_val = np.asarray([int(float(row["label"])) for row in val_rows], dtype=np.int64)

    all_feature_names = sorted(k for k in train_rows[0] if k not in ignore)
    old_feature_names = _old_feature_names(all_feature_names)
    feature_sets = {
        "all": all_feature_names,
        "old_118": old_feature_names,
        "meta": [
            k
            for k in all_feature_names
            if k.startswith("dx_")
            or k.startswith("item35")
            or k in {"side_right", "duration_s_manifest", "start_s_manifest"}
        ],
        "old_openmeta": [
            k
            for k in old_feature_names
            if k.startswith(("open_", "open_px_", "dx_", "item35"))
            or k in {"side_right", "duration_s_manifest"}
        ],
    }
    feature_sets = {name: features for name, features in feature_sets.items() if features}

    models = {
        "logreg_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3, multi_class="auto"),
        ),
        "svc_rbf_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(C=3.0, gamma="scale", class_weight="balanced"),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=1000,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=17,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=1000,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=23,
            ),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(random_state=29),
        ),
        "extra_trees_old_seed0": make_pipeline(
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
        "extra_trees_old_seed98": make_pipeline(
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
    }

    results: dict[str, Any] = {
        "feature_sets": feature_sets,
        "models": {},
        "ensembles": {},
    }
    best: dict[str, Any] | None = None
    for feature_set, feature_names in feature_sets.items():
        x_train = np.asarray([[_float(row.get(k)) for k in feature_names] for row in train_rows], dtype=np.float64)
        x_val = np.asarray([[_float(row.get(k)) for k in feature_names] for row in val_rows], dtype=np.float64)
        for name, model_template in models.items():
            model = clone(model_template)
            model.fit(x_train, y_train)
            pred = model.predict(x_val).astype(np.int64)
            acc = float(accuracy_score(y_val, pred))
            qwk = float(cohen_kappa_score(y_val, pred, weights="quadratic"))
            mae = float(mean_absolute_error(y_val, pred))
            entry = {
                "model": name,
                "feature_set": feature_set,
                "val_acc": acc,
                "qwk": qwk,
                "mae": mae,
                "confusion_matrix": confusion_matrix(y_val, pred, labels=[0, 1, 2, 3, 4]).tolist(),
                "predictions": pred.tolist(),
                "labels": y_val.tolist(),
            }
            results["models"][f"{feature_set}:{name}"] = entry
            if best is None or (acc, qwk, -mae) > (best["val_acc"], best["qwk"], -best["mae"]):
                best = entry

    ensemble_specs = {
        "validation_selected_median_vote_v1": [
            ("old_118", "extra_trees_old_seed0"),
            ("meta", "extra_trees_old_seed0"),
            ("old_openmeta", "random_forest"),
            ("meta", "gradient_boosting"),
        ],
    }
    for ensemble_name, members in ensemble_specs.items():
        member_keys = [f"{feature_set}:{model_name}" for feature_set, model_name in members]
        if not all(key in results["models"] for key in member_keys):
            continue
        member_entries = [results["models"][key] for key in member_keys]
        pred_matrix = np.asarray([entry["predictions"] for entry in member_entries], dtype=np.float64)
        pred = np.rint(np.median(pred_matrix, axis=0)).clip(0, 4).astype(np.int64)
        acc = float(accuracy_score(y_val, pred))
        qwk = float(cohen_kappa_score(y_val, pred, weights="quadratic"))
        mae = float(mean_absolute_error(y_val, pred))
        entry = {
            "model": ensemble_name,
            "feature_set": "ensemble",
            "ensemble_method": "median_vote",
            "members": member_keys,
            "member_correct": [
                int(np.sum(np.asarray(member["predictions"], dtype=np.int64) == y_val))
                for member in member_entries
            ],
            "val_acc": acc,
            "qwk": qwk,
            "mae": mae,
            "confusion_matrix": confusion_matrix(y_val, pred, labels=[0, 1, 2, 3, 4]).tolist(),
            "predictions": pred.tolist(),
            "labels": y_val.tolist(),
            "selection_note": "Selected on fold-0 validation predictions after model-family search.",
        }
        results["ensembles"][ensemble_name] = entry
        if best is None or (acc, qwk, -mae) > (best["val_acc"], best["qwk"], -best["mae"]):
            best = entry
    results["best"] = best
    return results


def _old_feature_names(feature_names: list[str]) -> list[str]:
    old_metrics = {
        "n",
        "mean",
        "std",
        "min",
        "max",
        "range",
        "p10",
        "p50",
        "p90",
        "slope",
        "first_mean",
        "last_mean",
        "decrement",
        "velocity_mean_abs",
        "velocity_max_abs",
        "peak_count",
        "peak_rate_hz",
        "dominant_freq_hz",
        "dominant_power",
    }
    series = ("bbox_area", "open", "open_px", "wrist_x", "wrist_y")
    keep = {
        "duration_s_manifest",
        "frame_count_manifest",
        "frames_seen",
        "frames_with_hand",
        "hand_detect_rate",
        "item35_other",
        "item35_same",
        "side_right",
        "start_s_manifest",
    }
    keep.update(f"dx_{dx}" for dx in ("HC", "NDC", "PD", "PPD"))
    for score in range(5):
        keep.add(f"item35_same_{score}")
        keep.add(f"item35_other_{score}")
    for prefix in series:
        keep.update(f"{prefix}_{metric}" for metric in old_metrics)
    return sorted(k for k in feature_names if k in keep)


def _float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        value = float(value)
        if math.isfinite(value):
            return value
        return np.nan
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
