#!/usr/bin/env python
"""Aggregate 5-fold leak-free hybrid stacking results.

Reads multiseed_oof_probs.json from each fold (fold 0 uses
multiseed_oof_probs_clean.json) and produces:
- per-fold mean QWK for kin+dx and kin+dx+V-JEPA-OOF
- across-fold mean ± std
- per-seed paired comparison aggregated across folds (Wilcoxon)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


HYBRID_ROOT = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4"
)
FOLD_FILES = {
    0: HYBRID_ROOT / "fold_0/oof_stack/multiseed_oof_probs_clean.json",
    1: HYBRID_ROOT / "fold_1/oof_stack/multiseed_oof_probs.json",
    2: HYBRID_ROOT / "fold_2/oof_stack/multiseed_oof_probs.json",
    3: HYBRID_ROOT / "fold_3/oof_stack/multiseed_oof_probs.json",
    4: HYBRID_ROOT / "fold_4/oof_stack/multiseed_oof_probs.json",
}

BASE_KEY = "kin_plus_dx"
HYB_KEY = "kin_plus_dx_plus_vjepa_all"


def main():
    rows = []
    base_per_seed = {}
    hyb_per_seed = {}
    for k, path in FOLD_FILES.items():
        if not path.exists():
            print(f"[fold {k}] MISSING: {path}")
            continue
        d = json.loads(path.read_text())
        b = d[BASE_KEY]
        h = d[HYB_KEY]
        b_q = np.array([r["qwk"] for r in b["per_seed"]])
        h_q = np.array([r["qwk"] for r in h["per_seed"]])
        base_per_seed[k] = b_q
        hyb_per_seed[k] = h_q
        delta = h_q - b_q
        print(
            f"[fold {k}] base {b_q.mean():.4f}±{b_q.std():.4f}  "
            f"hyb {h_q.mean():.4f}±{h_q.std():.4f}  "
            f"Δ={delta.mean():+.4f}  ({(delta>0).sum()}/{len(delta)})"
        )
        rows.append((k, b_q.mean(), b_q.std(), h_q.mean(), h_q.std(), delta.mean()))

    if not rows:
        print("no folds available yet")
        return

    print()
    folds_done = [r[0] for r in rows]
    base_means = np.array([r[1] for r in rows])
    hyb_means = np.array([r[3] for r in rows])
    print(f"=== across-fold ({len(rows)} folds: {folds_done}) ===")
    print(f"baseline  mean QWK across folds = {base_means.mean():.4f} ± {base_means.std():.4f}")
    print(f"hybrid    mean QWK across folds = {hyb_means.mean():.4f} ± {hyb_means.std():.4f}")
    print(f"Δ                               = {(hyb_means - base_means).mean():+.4f}")

    if len(rows) == 5:
        # All-folds-all-seeds aggregate Wilcoxon (50 paired observations)
        b_all = np.concatenate([base_per_seed[k] for k in [0, 1, 2, 3, 4]])
        h_all = np.concatenate([hyb_per_seed[k] for k in [0, 1, 2, 3, 4]])
        stat = wilcoxon(h_all, b_all, alternative="greater")
        print(f"\n=== pooled (50 fold×seed) ===")
        print(f"baseline  {b_all.mean():.4f} ± {b_all.std():.4f}")
        print(f"hybrid    {h_all.mean():.4f} ± {h_all.std():.4f}")
        print(f"Δ         {(h_all - b_all).mean():+.4f}  median={np.median(h_all - b_all):+.4f}")
        print(f"wins      {((h_all - b_all) > 0).sum()}/{len(b_all)}")
        print(f"Wilcoxon  p={stat.pvalue:.4f} (one-sided, hybrid > baseline)")

    # Dump aggregate for figure-making
    out = Path(
        "/gpfs/milgram/pi/scherzer/yl2428/vjepa2/docs/results/"
        "hybrid_5fold_aggregate.json"
    )
    payload = {
        "folds_done": folds_done,
        "per_fold": {
            str(k): {
                "base_qwk": base_per_seed[k].tolist(),
                "hyb_qwk": hyb_per_seed[k].tolist(),
            }
            for k in base_per_seed
        },
        "across_fold_base_mean": float(base_means.mean()),
        "across_fold_base_std": float(base_means.std()),
        "across_fold_hyb_mean": float(hyb_means.mean()),
        "across_fold_hyb_std": float(hyb_means.std()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
