#!/usr/bin/env python3
"""Train-split calibration for adaptive WindowMIL prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_adaptive/item_3_4"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--score", action="append", default=[], help="name=train_glob::val_glob")
    parser.add_argument("--include-default-windowmil", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--max-threshold-candidates", type=int, default=25)
    parser.add_argument("--blend-step", type=float, default=0.05)
    args = parser.parse_args()

    specs = _default_specs(args.root) if args.include_default_windowmil else {}
    for raw in args.score:
        name, paths = raw.split("=", 1)
        train_glob, val_glob = paths.split("::", 1)
        specs[name] = (train_glob, val_glob)
    if not specs:
        raise SystemExit("No scores requested. Use --include-default-windowmil or --score name=train::val.")

    loaded = {}
    for name, (train_glob, val_glob) in specs.items():
        try:
            loaded[name] = (
                _read_predictions(_one_match(train_glob)),
                _read_predictions(_one_match(val_glob)),
            )
        except FileNotFoundError as exc:
            print(f"skip {name}: {exc}")

    if not loaded:
        raise SystemExit("No prediction pairs were found.")

    train = _merge_split({name: pair[0] for name, pair in loaded.items()})
    val = _merge_split({name: pair[1] for name, pair in loaded.items()})
    names = list(loaded)
    metric_train = Metrics(train["label"].astype(int), args.num_classes)
    metric_val = Metrics(val["label"].astype(int), args.num_classes)

    results = []
    print(f"scores={names} train_n={len(train['label'])} val_n={len(val['label'])}")
    print("\nuncalibrated emitted val")
    for name in names:
        pred = val[f"{name}__pred"].astype(int)
        row = _row("emitted", name, metric_val, pred)
        results.append(row)
        _print_row(row)

    print("\nthresholds fit on train, evaluated on val")
    for name in names:
        train_score = train[f"{name}__score"]
        val_score = val[f"{name}__score"]
        thresholds, train_fit = _fit_thresholds(
            train_score,
            metric_train,
            max_candidates=args.max_threshold_candidates,
        )
        val_pred = _threshold_predict(val_score, thresholds)
        row = _row("threshold", name, metric_val, val_pred)
        row["thresholds"] = [float(x) for x in thresholds]
        row["train_fit"] = train_fit
        results.append(row)
        _print_row(row, suffix=f" th={[round(x, 4) for x in thresholds]}")

    if len(names) >= 2:
        print("\npairwise weighted score blends, thresholds fit on train")
    for left, right in itertools.combinations(names, 2):
        best = None
        for weight in np.arange(0.0, 1.0 + 0.5 * args.blend_step, args.blend_step):
            train_score = weight * train[f"{left}__score"] + (1.0 - weight) * train[f"{right}__score"]
            thresholds, train_fit = _fit_thresholds(
                train_score,
                metric_train,
                max_candidates=args.max_threshold_candidates,
            )
            val_score = weight * val[f"{left}__score"] + (1.0 - weight) * val[f"{right}__score"]
            val_pred = _threshold_predict(val_score, thresholds)
            row = _row("blend_threshold", f"{left}+{right}", metric_val, val_pred)
            row["weight_left"] = float(weight)
            row["thresholds"] = [float(x) for x in thresholds]
            row["train_fit"] = train_fit
            key = (row["qwk"], row["acc"], -row["mae"])
            if best is None or key > best[0]:
                best = (key, row)
        if best is not None:
            results.append(best[1])
            _print_row(best[1], suffix=f" w_left={best[1]['weight_left']:.2f}")

    summary = {
        "root": str(args.root),
        "scores": names,
        "train_n": int(len(train["label"])),
        "val_n": int(len(val["label"])),
        "results": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.output_json}")


def _default_specs(root: Path) -> dict[str, tuple[str, str]]:
    mapping = {
        "topk4": ("fold_0_windowmil_fixedbest_topk_trainonly", "fold_0_windowmil_fixedbest_topk_valonly"),
        "topk8": ("fold_0_windowmil_fixedbest_topk8_trainonly", "fold_0_windowmil_fixedbest_topk8_valonly"),
        "meanprob": ("fold_0_windowmil_fixedbest_meanprob_trainonly", "fold_0_windowmil_fixedbest_meanprob_valonly"),
        "meanlogit": ("fold_0_windowmil_fixedbest_meanlogit_trainonly", "fold_0_windowmil_fixedbest_meanlogit_valonly"),
        "maxprob": ("fold_0_windowmil_fixedbest_maxprob_trainonly", "fold_0_windowmil_fixedbest_maxprob_valonly"),
        "learnedstats": (
            "fold_0_windowmil_learnedstats_frozen_scorer_trainonly_evalbest",
            "fold_0_windowmil_learnedstats_frozen_scorer_valonly_evalbest",
        ),
    }
    return {
        name: (
            str(root / train_slug / "video_classification_frozen" / "*" / "predictions_val_e1.csv"),
            str(root / val_slug / "video_classification_frozen" / "*" / "predictions_val_e1.csv"),
        )
        for name, (train_slug, val_slug) in mapping.items()
    }


def _one_match(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def _read_predictions(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_best", "1") != "1":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    rows.sort(key=lambda row: int(row["sample_index"]))
    return {
        "sample_index": np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64),
        "label": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        "pred": np.asarray([int(row["pred"]) for row in rows], dtype=np.int64),
        "score": np.asarray([float(row["score"]) for row in rows], dtype=np.float64),
    }


def _merge_split(items: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    first_name = next(iter(items))
    merged = {
        "sample_index": items[first_name]["sample_index"],
        "label": items[first_name]["label"],
    }
    for name, item in items.items():
        if not np.array_equal(item["sample_index"], merged["sample_index"]):
            raise ValueError(f"sample_index mismatch for {name}")
        if not np.array_equal(item["label"], merged["label"]):
            raise ValueError(f"label mismatch for {name}")
        merged[f"{name}__pred"] = item["pred"]
        merged[f"{name}__score"] = item["score"]
    return merged


class Metrics:
    def __init__(self, labels: np.ndarray, num_classes: int) -> None:
        self.labels = labels.astype(int)
        self.num_classes = int(num_classes)
        self.classes = np.arange(self.num_classes)
        self.hist_true = np.bincount(self.labels, minlength=self.num_classes).astype(np.float64)

    def score(self, pred: np.ndarray) -> dict[str, object]:
        pred = pred.astype(int)
        return {
            "acc": float((pred == self.labels).mean()),
            "qwk": float(self.qwk(pred)),
            "mae": float(np.abs(pred - self.labels).mean()),
            "pred_counts": np.bincount(pred, minlength=self.num_classes).astype(int).tolist(),
        }

    def qwk(self, pred: np.ndarray) -> float:
        observed = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        np.add.at(observed, (self.labels, pred), 1.0)
        hist_pred = np.bincount(pred, minlength=self.num_classes).astype(np.float64)
        expected = np.outer(self.hist_true, hist_pred) / max(1, len(self.labels))
        weights = ((self.classes[:, None] - self.classes[None, :]) ** 2) / float((self.num_classes - 1) ** 2)
        denom = float((weights * expected).sum())
        if denom == 0.0:
            return 0.0
        return 1.0 - float((weights * observed).sum()) / denom


def _fit_thresholds(score: np.ndarray, metrics: Metrics, max_candidates: int) -> tuple[tuple[float, ...], dict[str, object]]:
    quantiles = np.linspace(0.03, 0.97, int(max_candidates))
    candidates = np.unique(np.quantile(score, quantiles))
    if len(candidates) < metrics.num_classes - 1:
        candidates = np.unique(np.linspace(score.min(), score.max(), int(max_candidates)))
    best_key = None
    best_thresholds = None
    best_metrics = None
    for thresholds in itertools.combinations(candidates, metrics.num_classes - 1):
        pred = _threshold_predict(score, thresholds)
        row = metrics.score(pred)
        key = (row["qwk"], row["acc"], -row["mae"])
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = tuple(float(x) for x in thresholds)
            best_metrics = row
    if best_thresholds is None or best_metrics is None:
        raise RuntimeError("threshold search failed")
    return best_thresholds, best_metrics


def _threshold_predict(score: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    return np.digitize(score, thresholds).clip(0, len(thresholds)).astype(int)


def _row(kind: str, name: str, metrics: Metrics, pred: np.ndarray) -> dict[str, object]:
    row = metrics.score(pred)
    row.update({"kind": kind, "name": name})
    return row


def _print_row(row: dict[str, object], suffix: str = "") -> None:
    print(
        f"{row['kind']:16s} {row['name']:30s} "
        f"acc={100.0 * row['acc']:.1f} qwk={row['qwk']:.3f} mae={row['mae']:.3f} "
        f"counts={row['pred_counts']}{suffix}"
    )


if __name__ == "__main__":
    main()
