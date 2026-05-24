#!/usr/bin/env python
"""Ensemble the top-K tstats configs by averaging val probabilities.

Reads `temporal_encoder_results.json` produced by train_temporal_encoder.py,
takes the top-K single-model rows ranked by QWK, averages their per-class
val probabilities, and reports the ensemble metrics.

This is a no-additional-training way to combine the seed/config diversity
already in the sweep results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", required=True,
                   help="Path to temporal_encoder_results.json from train_temporal_encoder.py")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--rank-by", default="qwk", choices=["qwk", "correct", "val_acc"])
    p.add_argument("--out-json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    d = json.loads(Path(args.results_json).read_text())
    top_results = d.get("top_results") or []
    if not top_results and "best" in d:
        top_results = [d["best"]]
    # Filter to rows with val_probs
    rows = [r for r in top_results if "val_probs" in r and r["val_probs"]]
    if not rows:
        raise SystemExit("No top_results rows with val_probs found.")

    rank_key = args.rank_by
    rows = sorted(rows, key=lambda r: -float(r.get(rank_key, 0)))
    pool = rows[: args.top]
    print(f"Ensembling top {len(pool)} of {len(rows)} configs")
    for i, r in enumerate(pool):
        cfg = r.get("config") or r.get("cfg") or {}
        print(f"  {i+1}. qwk={r.get('qwk',0):.4f} acc={r.get('val_acc', r.get('correct', 0)):.4f} "
              f"cfg=seed{cfg.get('seed')}_lr{cfg.get('lr')}_loss{cfg.get('loss')}_mix{cfg.get('mixup_alpha')}_fft{cfg.get('fft_bands')}")

    probs = np.array([r["val_probs"] for r in pool], dtype=np.float64)  # [K, N, C]
    avg_probs = probs.mean(axis=0)  # [N, C]
    labels = np.array(pool[0]["labels"], dtype=np.int64)
    # Verify all rows have same labels
    for r in pool[1:]:
        assert r["labels"] == pool[0]["labels"], "label mismatch across rows"
    yhat = avg_probs.argmax(axis=1)
    qwk = float(cohen_kappa_score(labels, yhat, weights="quadratic"))
    acc = float(accuracy_score(labels, yhat))
    mae = float(mean_absolute_error(labels, yhat))
    print(f"\nENSEMBLE TOP-{args.top} val: acc={acc:.4f} qwk={qwk:.4f} mae={mae:.4f}")
    # Also try expected-value rounding (ordinal regression style)
    classes = np.arange(probs.shape[-1])
    expected = (avg_probs * classes).sum(axis=1)
    yhat2 = np.clip(np.round(expected), 0, classes.max()).astype(np.int64)
    qwk2 = float(cohen_kappa_score(labels, yhat2, weights="quadratic"))
    acc2 = float(accuracy_score(labels, yhat2))
    mae2 = float(mean_absolute_error(labels, yhat2))
    print(f"ENSEMBLE expected-rounded: acc={acc2:.4f} qwk={qwk2:.4f} mae={mae2:.4f}")

    result = {
        "argmax": {"acc": acc, "qwk": qwk, "mae": mae},
        "expected_rounded": {"acc": acc2, "qwk": qwk2, "mae": mae2},
        "k": args.top,
        "pool_qwk": [r.get("qwk") for r in pool],
        "pool_configs": [r.get("config") or r.get("cfg") for r in pool],
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
