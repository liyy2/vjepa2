#!/usr/bin/env python3
"""OOF-calibrated stacker for cached V-JEPA temporal-stat models.

The cross-sweep ensemble uses held-out fold-0 validation labels to pick members.
This script reruns a small selected set of cached temporal-stat configs inside
fold-0 train, builds out-of-fold train probabilities, fits simple score
calibrators on those OOF predictions only, and evaluates the stored full-train
validation probabilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pd_hand import train_temporal_encoder as te


DEFAULT_CROSS = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/"
    "cross_sweep_ensemble_refresh_20260521.json"
)
DEFAULT_OUT = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/"
    "temporal_oof_stacker_top_greedy.json"
)
DEFAULT_SPLIT_DIR = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/splits/item_3_4"
)
NUM_CLASSES = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-sweep", type=Path, default=DEFAULT_CROSS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--select", choices=["greedy", "top", "union"], default="union")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank-by", choices=["qwk", "acc"], default="qwk")
    parser.add_argument("--max-threshold-candidates", type=int, default=25)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--inner-split-mode", choices=["subject", "stratified"], default="subject")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_SPLIT_DIR / "fold_0_train.csv")
    parser.add_argument("--val-csv", type=Path, default=DEFAULT_SPLIT_DIR / "fold_0_val.csv")
    parser.add_argument("--save-probs", action="store_true",
                        help="Write a companion NPZ with raw candidate OOF/validation probabilities.")
    args = parser.parse_args()

    te._RANK_BY = args.rank_by
    cross = json.loads(args.cross_sweep.read_text())
    items = gather_items(cross["candidate_sources"])
    selected = select_items(items, cross, mode=args.select, top_n=args.top_n, max_k=args.max_k)
    print(f"loaded_candidates={len(items)} selected={len(selected)}", flush=True)
    for idx, item in enumerate(selected):
        cfg = item["config"]
        print(
            f"selected[{idx}] qwk={item['qwk']:.4f} acc={item['val_acc']:.4f} "
            f"seed={cfg.get('seed')} loss={cfg.get('loss')} mixup={cfg.get('mixup_alpha')} "
            f"cw={cfg.get('class_weight')} fft={cfg.get('fft_bands')} vel={cfg.get('use_velocity')} "
            f"source={Path(item['source']).parent.name}",
            flush=True,
        )

    oof_probs, val_probs, labels_train, labels_val, candidate_meta = build_oof_predictions(args, selected)
    results = evaluate_stackers(
        oof_probs=oof_probs,
        val_probs=val_probs,
        y_train=labels_train,
        y_val=labels_val,
        candidate_meta=candidate_meta,
        max_threshold_candidates=args.max_threshold_candidates,
        blend_step=args.blend_step,
    )
    results_sorted_by_oof = sorted(results, key=lambda r: (r["oof_qwk"], r["oof_acc"], -r["oof_mae"]), reverse=True)
    results_sorted_by_val = sorted(results, key=lambda r: (r["val_qwk"], r["val_acc"], -r["val_mae"]), reverse=True)

    print("\nTop selections by OOF train metric, evaluated on held-out val:")
    for row in results_sorted_by_oof[:12]:
        print(format_row(row), flush=True)
    print("\nTop diagnostic rows by val metric:")
    for row in results_sorted_by_val[:8]:
        print(format_row(row), flush=True)

    output = {
        "note": (
            "OOF train-calibrated stacker. Selection, blend weights, and thresholds "
            "are fit on fold-0 train OOF predictions; held-out fold-0 val labels are "
            "used only for final evaluation."
        ),
        "cross_sweep": str(args.cross_sweep),
        "selected_count": len(selected),
        "inner_splits": int(args.inner_splits),
        "inner_split_mode": args.inner_split_mode,
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "train_n": int(labels_train.size),
        "val_n": int(labels_val.size),
        "selected_candidates": candidate_meta,
        "best_by_oof": results_sorted_by_oof[0],
        "best_by_val_diagnostic": results_sorted_by_val[0],
        "results": results_sorted_by_oof,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    if args.save_probs:
        prob_path = args.out.with_suffix(".probs.npz")
        np.savez_compressed(
            prob_path,
            oof_probs=oof_probs.astype(np.float32),
            val_probs=val_probs.astype(np.float32),
            labels_train=labels_train.astype(np.int64),
            labels_val=labels_val.astype(np.int64),
            candidate_sources=np.asarray([row["source"] for row in candidate_meta], dtype=object),
            candidate_configs=np.asarray([json.dumps(row["config"], sort_keys=True) for row in candidate_meta], dtype=object),
            selected_count=np.asarray([len(selected)], dtype=np.int64),
            inner_split_mode=np.asarray([args.inner_split_mode], dtype=object),
        )
        print(f"wrote {prob_path}", flush=True)


def gather_items(source_paths: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in source_paths:
        source_path = Path(source)
        data = json.loads(source_path.read_text())
        for result in data.get("top_results", []):
            if "val_probs" not in result:
                continue
            item = {
                "source": str(source_path),
                "source_train_npz": data["train_npz"],
                "source_val_npz": data["val_npz"],
                "pca_dim": int(data.get("pca_dim", 128)),
                "sample_balanced_normalization": bool(data.get("sample_balanced_normalization", False)),
                "pca_tokens_per_sample": int(data.get("pca_tokens_per_sample", 0)),
                "config": result["config"],
                "qwk": float(result["qwk"]),
                "val_acc": float(result["val_acc"]),
                "mae": float(result["mae"]),
                "val_probs": np.asarray(result["val_probs"], dtype=np.float32),
                "labels": np.asarray(result["labels"], dtype=np.int64),
            }
            items.append(item)
    if not items:
        raise SystemExit("No candidate val_probs found.")
    ref_labels = items[0]["labels"]
    for item in items[1:]:
        if not np.array_equal(item["labels"], ref_labels):
            raise SystemExit(f"Validation labels disagree: {item['source']}")
    items.sort(key=lambda row: row["qwk"], reverse=True)
    return items


def select_items(items: list[dict[str, Any]], cross: dict[str, Any], mode: str, top_n: int, max_k: int) -> list[dict[str, Any]]:
    selected_indices: list[int] = []
    if mode in {"top", "union"}:
        selected_indices.extend(range(min(top_n, len(items))))
    if mode in {"greedy", "union"}:
        pool_limit = min(max_k, len(items))
        for idx in cross.get("greedy_chosen_indices", []):
            if 0 <= int(idx) < pool_limit:
                selected_indices.append(int(idx))
    unique_indices = []
    seen = set()
    for idx in selected_indices:
        key = (
            items[idx]["source"],
            json.dumps(items[idx]["config"], sort_keys=True),
        )
        if key not in seen:
            seen.add(key)
            unique_indices.append(idx)
    return [items[idx] for idx in unique_indices]


def build_oof_predictions(args: argparse.Namespace, selected: list[dict[str, Any]]):
    labels_train_ref = None
    labels_val_ref = selected[0]["labels"]
    train_meta_rows = None
    subject_groups = None
    oof_list = []
    val_list = []
    meta = []
    for cand_idx, item in enumerate(selected):
        cfg = te.TrainConfig(**{k: item["config"][k] for k in asdict(te.TrainConfig(0, 1e-3, 1e-2, 0.5)).keys()})
        x_train_raw, y_train, mask_train = te.load_npz(item["source_train_npz"])
        if labels_train_ref is None:
            labels_train_ref = y_train
            train_meta_rows, subject_groups = _split_rows_in_cache_order(args.train_csv, y_train)
            if args.inner_split_mode == "subject":
                print(
                    f"inner_split_mode=subject unique_subjects={len(set(subject_groups.tolist()))}",
                    flush=True,
                )
        elif not np.array_equal(labels_train_ref, y_train):
            raise SystemExit(f"Train labels disagree for {item['source']}")
        if not np.array_equal(labels_val_ref, item["labels"]):
            raise SystemExit(f"Val labels disagree for {item['source']}")

        assert subject_groups is not None
        if args.inner_split_mode == "subject":
            splitter = StratifiedGroupKFold(n_splits=args.inner_splits, shuffle=True, random_state=1000 + cand_idx)
            split_iter = splitter.split(np.zeros_like(y_train), y_train, subject_groups)
        else:
            splitter = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=1000 + cand_idx)
            split_iter = splitter.split(np.zeros_like(y_train), y_train)
        oof = np.zeros((y_train.shape[0], NUM_CLASSES), dtype=np.float32)
        print(f"\nOOF candidate {cand_idx + 1}/{len(selected)} source={item['source']}", flush=True)
        for fold_idx, (fit_idx, hold_idx) in enumerate(split_iter, start=1):
            if args.inner_split_mode == "subject":
                fit_subjects = len(set(subject_groups[fit_idx].tolist()))
                hold_subjects = len(set(subject_groups[hold_idx].tolist()))
                print(
                    f"  inner_fold={fold_idx} fit={len(fit_idx)} hold={len(hold_idx)} "
                    f"subjects={fit_subjects}/{hold_subjects}",
                    flush=True,
                )
            else:
                print(f"  inner_fold={fold_idx} fit={len(fit_idx)} hold={len(hold_idx)}", flush=True)
            x_fit, x_hold = te.normalize_from_train(
                x_train_raw[fit_idx],
                x_train_raw[hold_idx],
                mask_train[fit_idx],
                mask_train[hold_idx],
                sample_balanced=item["sample_balanced_normalization"],
            )
            if item["pca_dim"] > 0:
                x_fit, x_hold = te.temporal_pca_from_train(
                    x_fit,
                    x_hold,
                    mask_train[fit_idx],
                    mask_train[hold_idx],
                    item["pca_dim"],
                    tokens_per_sample=item["pca_tokens_per_sample"],
                )
            result = te.train_one_config(
                cfg=cfg,
                x_train=x_fit,
                y_train=y_train[fit_idx],
                mask_train=mask_train[fit_idx],
                x_val=x_hold,
                y_val=y_train[hold_idx],
                mask_val=mask_train[hold_idx],
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                patience=args.patience,
            )
            oof[hold_idx] = np.asarray(result["val_probs"], dtype=np.float32)
            print(
                f"    fold_acc={result['val_acc']:.4f} fold_qwk={result['qwk']:.4f} "
                f"epoch={result['epoch']}",
                flush=True,
            )
        oof_list.append(oof)
        val_list.append(item["val_probs"])
        meta.append({
            "source": item["source"],
            "train_npz": item["source_train_npz"],
            "val_npz": item["source_val_npz"],
            "single_val_qwk": item["qwk"],
            "single_val_acc": item["val_acc"],
            "single_val_mae": item["mae"],
            "pca_dim": item["pca_dim"],
            "sample_balanced_normalization": item["sample_balanced_normalization"],
            "pca_tokens_per_sample": item["pca_tokens_per_sample"],
            "config": item["config"],
        })
    assert labels_train_ref is not None
    return (
        np.stack(oof_list, axis=0),
        np.stack(val_list, axis=0),
        labels_train_ref.astype(np.int64),
        labels_val_ref.astype(np.int64),
        meta,
    )


def _split_rows_in_cache_order(csv_path: Path, labels: np.ndarray) -> tuple[list[dict[str, str]], np.ndarray]:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(labels.size):
        raise SystemExit(f"{csv_path} has {len(rows)} rows, but cached labels have {labels.size}")
    order = _distributed_sampler_order(len(rows))
    ordered_rows = [rows[idx] for idx in order]
    ordered_labels = np.asarray([int(float(row["label"])) for row in ordered_rows], dtype=np.int64)
    if not np.array_equal(ordered_labels, labels):
        raise SystemExit(
            f"{csv_path} labels do not match cached NPZ order. "
            "The cache ordering assumption is wrong; do not use subject OOF results."
        )
    groups = np.asarray([row["subject_id"] for row in ordered_rows], dtype=object)
    return ordered_rows, groups


def _distributed_sampler_order(n_rows: int) -> list[int]:
    # cache_vjepa21_temporal_embeddings.py uses DistributedSampler with world_size=1,
    # rank=0, shuffle=True, and epoch=0. Reproduce that order so split metadata lines
    # up with older NPZ files that did not store per-sample coverage rows.
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.randperm(n_rows, generator=generator).tolist()


def evaluate_stackers(
    oof_probs: np.ndarray,
    val_probs: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    candidate_meta: list[dict[str, Any]],
    max_threshold_candidates: int,
    blend_step: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    n_candidates = oof_probs.shape[0]

    for idx in range(n_candidates):
        name = f"single_{idx}"
        results.append(_eval_probs(name, "single_argmax", oof_probs[idx], val_probs[idx], y_train, y_val, [idx]))
        results.append(_eval_thresholded_score(
            name,
            "single_score_threshold",
            _expected_score(oof_probs[idx]),
            _expected_score(val_probs[idx]),
            y_train,
            y_val,
            [idx],
            max_threshold_candidates,
        ))

    greedy = _greedy_by_oof(oof_probs, y_train)
    for prefix_len in range(1, len(greedy) + 1):
        indices = greedy[:prefix_len]
        oof_avg = oof_probs[indices].mean(axis=0)
        val_avg = val_probs[indices].mean(axis=0)
        results.append(_eval_probs(f"greedy_k{prefix_len}", "greedy_argmax", oof_avg, val_avg, y_train, y_val, indices))
        results.append(_eval_thresholded_score(
            f"greedy_k{prefix_len}",
            "greedy_score_threshold",
            _expected_score(oof_avg),
            _expected_score(val_avg),
            y_train,
            y_val,
            indices,
            max_threshold_candidates,
        ))

    for k in range(1, n_candidates + 1):
        indices = list(range(k))
        oof_avg = oof_probs[indices].mean(axis=0)
        val_avg = val_probs[indices].mean(axis=0)
        results.append(_eval_probs(f"top{k}", "topk_argmax", oof_avg, val_avg, y_train, y_val, indices))
        results.append(_eval_thresholded_score(
            f"top{k}",
            "topk_score_threshold",
            _expected_score(oof_avg),
            _expected_score(val_avg),
            y_train,
            y_val,
            indices,
            max_threshold_candidates,
        ))

    for left in range(n_candidates):
        for right in range(left + 1, n_candidates):
            for weight in np.arange(0.0, 1.0 + 0.5 * blend_step, blend_step):
                oof_score = weight * _expected_score(oof_probs[left]) + (1.0 - weight) * _expected_score(oof_probs[right])
                val_score = weight * _expected_score(val_probs[left]) + (1.0 - weight) * _expected_score(val_probs[right])
                row = _eval_thresholded_score(
                    f"pair_{left}_{right}",
                    "pair_score_threshold",
                    oof_score,
                    val_score,
                    y_train,
                    y_val,
                    [left, right],
                    max_threshold_candidates,
                )
                row["weight_left"] = float(weight)
                results.append(row)

    for row in results:
        row["candidate_summaries"] = [
            {
                "idx": int(idx),
                "single_val_qwk": candidate_meta[idx]["single_val_qwk"],
                "source": candidate_meta[idx]["source"],
                "config": candidate_meta[idx]["config"],
            }
            for idx in row["indices"]
        ]
    return results


def _greedy_by_oof(oof_probs: np.ndarray, y_train: np.ndarray) -> list[int]:
    chosen: list[int] = []
    remaining = set(range(oof_probs.shape[0]))
    running = np.zeros_like(oof_probs[0])
    best_seen = -1.0
    stale = 0
    while remaining:
        best_idx = None
        best_key = None
        for idx in sorted(remaining):
            avg = (running + oof_probs[idx]) / (len(chosen) + 1)
            m = _metrics(y_train, avg.argmax(axis=1).astype(np.int64))
            key = (m["qwk"], m["acc"], -m["mae"])
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
        assert best_idx is not None and best_key is not None
        running += oof_probs[best_idx]
        chosen.append(best_idx)
        remaining.remove(best_idx)
        if best_key[0] > best_seen:
            best_seen = best_key[0]
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    return chosen


def _eval_probs(name: str, kind: str, oof_prob: np.ndarray, val_prob: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, indices: list[int]):
    oof_pred = oof_prob.argmax(axis=1).astype(np.int64)
    val_pred = val_prob.argmax(axis=1).astype(np.int64)
    return _row(name, kind, y_train, oof_pred, y_val, val_pred, indices)


def _eval_thresholded_score(
    name: str,
    kind: str,
    oof_score: np.ndarray,
    val_score: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    indices: list[int],
    max_threshold_candidates: int,
):
    thresholds, oof_pred = _fit_thresholds(oof_score, y_train, max_threshold_candidates)
    val_pred = _threshold_predict(val_score, thresholds)
    row = _row(name, kind, y_train, oof_pred, y_val, val_pred, indices)
    row["thresholds"] = [float(x) for x in thresholds]
    return row


def _row(name: str, kind: str, y_train: np.ndarray, oof_pred: np.ndarray, y_val: np.ndarray, val_pred: np.ndarray, indices: list[int]):
    oof = _metrics(y_train, oof_pred)
    val = _metrics(y_val, val_pred)
    return {
        "name": name,
        "kind": kind,
        "indices": [int(i) for i in indices],
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


def _metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    acc, qwk, mae = _fast_metric_values(labels, pred)
    return {
        "acc": acc,
        "qwk": qwk,
        "mae": mae,
        "counts": np.bincount(pred, minlength=NUM_CLASSES).astype(int).tolist(),
        "confusion": confusion_matrix(labels, pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def _fast_metric_values(labels: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    labels = labels.astype(np.int64, copy=False)
    pred = pred.astype(np.int64, copy=False)
    acc = float((pred == labels).mean())
    mae = float(np.abs(pred - labels).mean())
    observed = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    np.add.at(observed, (labels, pred), 1.0)
    hist_true = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    hist_pred = np.bincount(pred, minlength=NUM_CLASSES).astype(np.float64)
    expected = np.outer(hist_true, hist_pred) / max(1, labels.size)
    classes = np.arange(NUM_CLASSES, dtype=np.float64)
    weights = ((classes[:, None] - classes[None, :]) ** 2) / float((NUM_CLASSES - 1) ** 2)
    denom = float((weights * expected).sum())
    if denom <= 0:
        qwk = 0.0
    else:
        qwk = 1.0 - float((weights * observed).sum()) / denom
    if not np.isfinite(qwk):
        qwk = 0.0
    return acc, float(qwk), mae


def _expected_score(probs: np.ndarray) -> np.ndarray:
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(denom, 1e-8), out=np.zeros_like(probs), where=denom > 0)
    return probs @ np.arange(NUM_CLASSES, dtype=np.float32)


def _fit_thresholds(score: np.ndarray, labels: np.ndarray, max_candidates: int):
    score = np.nan_to_num(score, nan=float(np.nanmean(score)) if np.isfinite(np.nanmean(score)) else 0.0)
    candidates = np.unique(np.quantile(score, np.linspace(0.03, 0.97, int(max_candidates))))
    if candidates.size < NUM_CLASSES - 1:
        candidates = np.unique(np.linspace(float(score.min()), float(score.max()), int(max_candidates)))
    if candidates.size < NUM_CLASSES - 1:
        center = float(score.mean())
        candidates = center + np.linspace(-2e-3, 2e-3, int(max_candidates))
    best_key = None
    best_thresholds = None
    best_pred = None
    import itertools
    for thresholds in itertools.combinations(candidates, NUM_CLASSES - 1):
        pred = _threshold_predict(score, thresholds)
        acc, qwk, mae = _fast_metric_values(labels, pred)
        key = (qwk, acc, -mae)
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = tuple(float(x) for x in thresholds)
            best_pred = pred
    if best_thresholds is None or best_pred is None:
        raise RuntimeError("threshold fitting failed")
    return best_thresholds, best_pred


def _threshold_predict(score: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    score = np.nan_to_num(score, nan=float(np.nanmean(score)) if np.isfinite(np.nanmean(score)) else 0.0)
    return np.digitize(score, thresholds).clip(0, NUM_CLASSES - 1).astype(np.int64)


def format_row(row: dict[str, Any]) -> str:
    extra = ""
    if "weight_left" in row:
        extra += f" w_left={row['weight_left']:.2f}"
    if "thresholds" in row:
        extra += " th=" + ",".join(f"{x:.3f}" for x in row["thresholds"])
    return (
        f"{row['kind']:24s} {row['name']:14s} idx={row['indices']} "
        f"oof={100.0 * row['oof_acc']:.1f}/{row['oof_qwk']:.3f}/{row['oof_mae']:.3f} "
        f"val={100.0 * row['val_acc']:.1f}/{row['val_qwk']:.3f}/{row['val_mae']:.3f}{extra}"
    )


if __name__ == "__main__":
    main()
