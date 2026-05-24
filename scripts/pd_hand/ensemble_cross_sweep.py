#!/usr/bin/env python
"""Cross-sweep ensemble: combine top configs across multiple sweep directories,
averaging val_probs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dirs", nargs="+")
    p.add_argument("--top-per-sweep", type=int, default=5)
    p.add_argument("--top-overall", type=int, action="append", default=None,
                   help="Top-K values for the cross-sweep pool")
    return p.parse_args()


def main():
    args = parse_args()
    if args.top_overall is None:
        args.top_overall = [3, 5, 8, 10, 15, 20, 30, 50]

    runs = []
    for d in args.sweep_dirs:
        for r in sorted(Path(d).glob("**/result.json")):
            try:
                x = json.loads(r.read_text())
            except Exception:
                continue
            best = x.get("best", {})
            metrics = best.get("metrics") or {}
            probs = metrics.get("val_probs")
            labels = metrics.get("val_labels") or best.get("val_labels")
            if probs is None or labels is None:
                continue
            runs.append({
                "tag": str(r.parent),
                "qwk": float(best.get("qwk", 0)),
                "probs": np.asarray(probs, dtype=np.float64),
                "labels": np.asarray(labels, dtype=np.int64),
            })
    if not runs:
        raise SystemExit("no runs found")
    runs.sort(key=lambda r: -r["qwk"])
    labels = runs[0]["labels"]
    for r in runs[1:]:
        if not np.array_equal(r["labels"], labels):
            raise SystemExit("label order mismatch across runs")
    print(f"pool size = {len(runs)} runs; best single = {runs[0]['qwk']:.4f}")
    print(f"\n{'k':>4} {'qwk':>8} {'acc':>8} {'mae':>8}")
    for k in args.top_overall:
        if k > len(runs):
            continue
        avg = np.mean([r["probs"] for r in runs[:k]], axis=0)
        yhat = avg.argmax(axis=1)
        qwk = cohen_kappa_score(labels, yhat, weights="quadratic")
        acc = accuracy_score(labels, yhat)
        mae = mean_absolute_error(labels, yhat)
        print(f"{k:>4} {qwk:>8.4f} {acc:>8.4f} {mae:>8.4f}")


if __name__ == "__main__":
    main()
