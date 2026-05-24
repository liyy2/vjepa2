#!/usr/bin/env python
"""Ensemble combiner sweep runs by averaging val_probs.

Walks a sweep directory, picks runs whose `result.json` has `best.metrics.val_probs`,
sorts by val QWK, takes top-K, averages probabilities, reports the ensemble's QWK / acc / MAE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir")
    p.add_argument("--top", action="append", type=int, default=None,
                   help="Top-K values to try (pass multiple times)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.top is None:
        args.top = [1, 2, 3, 5, 8, 10, 15, 20]

    runs = []
    for r in sorted(Path(args.sweep_dir).glob("**/result.json")):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        best = d.get("best", {})
        metrics = best.get("metrics") or {}
        probs = metrics.get("val_probs")
        labels = metrics.get("val_labels") or best.get("val_labels")
        if probs is None or labels is None:
            continue
        runs.append({
            "tag": str(r.parent.relative_to(args.sweep_dir)),
            "qwk": float(best.get("qwk", 0)),
            "acc": float(best.get("acc", 0)),
            "probs": np.asarray(probs, dtype=np.float64),
            "labels": np.asarray(labels, dtype=np.int64),
            "n_val": int(d.get("n_val", 0)),
        })
    if not runs:
        raise SystemExit(f"No result.json with val_probs found under {args.sweep_dir}")
    runs.sort(key=lambda r: -r["qwk"])

    # Verify labels consistent
    labels = runs[0]["labels"]
    for r in runs[1:]:
        if not np.array_equal(r["labels"], labels):
            raise SystemExit("Label order mismatch between runs (cache reshuffle?)")
    print(f"pool size: {len(runs)}; n_val={len(labels)}; best single QWK={runs[0]['qwk']:.4f}")
    print(f"\n{'k':>4} {'qwk':>8} {'acc':>8} {'mae':>8}")
    for k in args.top:
        if k > len(runs):
            continue
        avg = np.mean([r["probs"] for r in runs[:k]], axis=0)
        yhat = avg.argmax(axis=1)
        qwk = cohen_kappa_score(labels, yhat, weights="quadratic")
        acc = accuracy_score(labels, yhat)
        mae = mean_absolute_error(labels, yhat)
        flag = " *" if k == 1 else ""
        print(f"{k:>4} {qwk:>8.4f} {acc:>8.4f} {mae:>8.4f}{flag}")


if __name__ == "__main__":
    main()
