#!/usr/bin/env python
"""Build the best ensemble across cached-feature temporal-encoder sweeps.

Loads stored val_probs from every sweep's top-K configs, then searches for the
ensemble (subset of configs averaged) that maximizes QWK on val.

Two search modes:
  - greedy: forward selection. Start empty; at each step add the config that
    most improves QWK on the running average.
  - top_k_by_qwk: just average the top-K configs (sorted by single-config QWK)
    for K = 1, 2, ..., 50.

Each sweep directory is expected to contain shard*/temporal_encoder_results.json
files produced by `train_temporal_encoder.py`. Reports the best QWK / acc / mae.

Example:
  python cross_sweep_ensemble.py \\
      --sweeps <sweeps>/fold_0_velocity <sweeps>/fold_0_mega_seeds \\
      --out cross_sweep.json
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", nargs="+", required=True,
                   help="Paths to either an individual results.json or a directory containing shard*/temporal_encoder_results.json")
    p.add_argument("--out", default="/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/cross_sweep_ensemble.json")
    p.add_argument("--max-K", type=int, default=80, help="Cap on candidate pool size (sorted by QWK)")
    p.add_argument("--target-qwk", type=float, default=0.79, help="Kinematic baseline to beat")
    p.add_argument("--target-acc", type=float, default=0.682)
    return p.parse_args()


def gather_results(paths):
    """Return list of (qwk, val_probs, labels, source, config). Uses partial files when no final is present."""
    items = []
    for p in paths:
        p = Path(p)
        files = []
        if p.is_dir():
            # Prefer final temporal_encoder_results.json per shard; fall back to partial
            for shard_dir in sorted(p.glob("shard*")):
                final = shard_dir / "temporal_encoder_results.json"
                partial = shard_dir / "temporal_encoder_partial_results.json"
                if final.exists():
                    files.append(final)
                elif partial.exists():
                    files.append(partial)
            top_final = p / "temporal_encoder_results.json"
            top_partial = p / "temporal_encoder_partial_results.json"
            if top_final.exists():
                files.append(top_final)
            elif top_partial.exists():
                files.append(top_partial)
        elif p.is_file():
            files.append(p)
        for f in files:
            try:
                d = json.load(open(f))
            except Exception:
                continue
            top = d.get("top_results", [])
            for r in top:
                if "val_probs" not in r:
                    continue
                items.append({
                    "source": str(f),
                    "config": r["config"],
                    "qwk": r["qwk"],
                    "val_acc": r["val_acc"],
                    "mae": r["mae"],
                    "val_probs": np.asarray(r["val_probs"], dtype=np.float32),
                    "labels": np.asarray(r["labels"], dtype=np.int64),
                })
    if not items:
        raise SystemExit("no results found in given sweeps")
    # Sanity: all label vectors must match
    ref_labels = items[0]["labels"]
    for it in items[1:]:
        if not np.array_equal(it["labels"], ref_labels):
            raise SystemExit("label vectors disagree across sweeps — different val set?")
    # Sort by single-config QWK desc
    items.sort(key=lambda x: x["qwk"], reverse=True)
    return items, ref_labels


def score(labels, probs):
    pred = probs.argmax(axis=1).astype(np.int64)
    return {
        "qwk": float(cohen_kappa_score(labels, pred, weights="quadratic")),
        "val_acc": float(accuracy_score(labels, pred)),
        "mae": float(mean_absolute_error(labels, pred)),
        "n": int(labels.size),
        "correct": int((pred == labels).sum()),
        "confusion": confusion_matrix(labels, pred, labels=[0, 1, 2, 3, 4]).tolist(),
        "pred": pred.tolist(),
    }


def topk_average_scan(items, labels, K_max):
    best = None
    out = []
    running = np.zeros_like(items[0]["val_probs"])
    for k, it in enumerate(items[:K_max], start=1):
        running += it["val_probs"]
        avg = running / k
        m = score(labels, avg)
        m["K"] = k
        out.append(m)
        if best is None or m["qwk"] > best["qwk"]:
            best = m
    return out, best


def greedy_forward(items, labels, pool_size, target_qwk):
    pool = items[:pool_size]
    chosen = []
    remaining = set(range(len(pool)))
    running = np.zeros_like(pool[0]["val_probs"])
    best_score = None
    history = []
    while remaining:
        candidate = None
        for idx in remaining:
            trial_avg = (running + pool[idx]["val_probs"]) / (len(chosen) + 1)
            sc = score(labels, trial_avg)
            if candidate is None or sc["qwk"] > candidate[1]["qwk"]:
                candidate = (idx, sc)
        idx, sc = candidate
        running += pool[idx]["val_probs"]
        chosen.append(idx)
        remaining.remove(idx)
        sc["K"] = len(chosen)
        sc["just_added_qwk"] = pool[idx]["qwk"]
        sc["just_added_cfg"] = pool[idx]["config"]
        history.append(sc)
        if best_score is None or sc["qwk"] > best_score["qwk"]:
            best_score = sc
        # Stop early if we plateau for 8 consecutive adds
        if len(history) > 8 and all(h["qwk"] <= best_score["qwk"] for h in history[-8:]):
            break
    return history, best_score, chosen, pool


def main():
    args = parse_args()
    items, labels = gather_results(args.sweeps)
    print(f"Loaded {len(items)} stored val_probs from {len({it['source'] for it in items})} files")
    print(f"Top 8 single-model QWK:")
    for it in items[:8]:
        c = it["config"]
        print(f"  QWK={it['qwk']:.4f} acc={it['val_acc']:.4f} | {c.get('model_type','?')} loss={c.get('loss','?')} mixup={c.get('mixup_alpha','?')} cw={c.get('class_weight','?')} drop={c.get('dropout','?')} hd={c.get('hidden_dim','?')} bins={c.get('temporal_bins','?')} fft={c.get('fft_bands','?')} fft_mode={c.get('fft_mode','?')} vel={c.get('use_velocity','?')} seed={c.get('seed','?')}")

    # Top-K average scan
    scan, best_topk = topk_average_scan(items, labels, K_max=args.max_K)
    print(f"\nBest top-K average: K={best_topk['K']} QWK={best_topk['qwk']:.4f} acc={best_topk['val_acc']:.4f} mae={best_topk['mae']:.4f}")

    # Greedy forward selection
    history, best_greedy, chosen_idx, pool = greedy_forward(items, labels, pool_size=args.max_K, target_qwk=args.target_qwk)
    print(f"Best greedy ensemble: K={best_greedy['K']} QWK={best_greedy['qwk']:.4f} acc={best_greedy['val_acc']:.4f} mae={best_greedy['mae']:.4f}")
    print(f"  beats kinematic baseline (QWK > {args.target_qwk})? {best_greedy['qwk'] > args.target_qwk}")
    print(f"  beats kinematic baseline (acc > {args.target_acc})? {best_greedy['val_acc'] > args.target_acc}")

    out = {
        "n_candidates": len(items),
        "candidate_sources": list(OrderedDict.fromkeys(it["source"] for it in items)),
        "top_single": [
            {"qwk": it["qwk"], "val_acc": it["val_acc"], "mae": it["mae"],
             "config": it["config"], "source": it["source"]}
            for it in items[:30]
        ],
        "topk_scan": scan,
        "greedy_best": best_greedy,
        "greedy_chosen_indices": chosen_idx,
        "greedy_chosen_configs": [pool[i]["config"] for i in chosen_idx],
        "target_qwk": args.target_qwk,
        "target_acc": args.target_acc,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
