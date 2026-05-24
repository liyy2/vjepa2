#!/usr/bin/env python
"""Walk the may19 item_3_4 fold_0 result directories and emit a sorted leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19"
)


def kin_row() -> list[dict]:
    p = BASE / "kinematic_baselines/item_3_4/fold_0/results_stride2.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    b = d.get("best", {})
    return [{
        "approach": "kinematic LightGBM (ExtraTrees old_118)",
        "qwk": float(b.get("qwk", 0)),
        "acc": float(b.get("val_acc", 0)),
        "mae": float(b.get("mae", 0)),
        "n_val": len(b.get("labels", [])),
        "where": str(p),
    }]


def combiner_dir_rows(root: Path, tag: str) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for r in sorted(root.glob("**/result.json")):
        try:
            d = json.loads(r.read_text())
        except Exception:
            continue
        best = d.get("best", {})
        rows.append({
            "approach": f"{tag}/{r.parent.relative_to(root)}",
            "qwk": float(best.get("qwk", 0)),
            "acc": float(best.get("acc", 0)),
            "mae": float((best.get("metrics") or {}).get("mae", 0)),
            "n_val": int(d.get("n_val", 0)),
            "where": str(r),
        })
    return rows


def tstats_rows(json_path: Path, tag: str) -> list[dict]:
    if not json_path.exists():
        return []
    d = json.loads(json_path.read_text())
    rows = []
    bsm = d.get("best_single_model")
    if bsm:
        rows.append({
            "approach": f"{tag} best single",
            "qwk": float(bsm.get("qwk", 0)),
            "acc": float(bsm.get("val_acc", 0)),
            "mae": float(bsm.get("mae", 0)),
            "n_val": len(bsm.get("labels", [])),
            "where": str(json_path),
        })
    be = d.get("best_ensemble")
    if be:
        rows.append({
            "approach": f"{tag} best ensemble (top-{be.get('ensemble_top_k') or be.get('config',{}).get('top_k')})",
            "qwk": float(be.get("qwk", 0)),
            "acc": float(be.get("val_acc", 0)),
            "mae": float(be.get("mae", 0)),
            "n_val": len(be.get("labels", [])),
            "where": str(json_path),
        })
    return rows


def main():
    rows = []
    rows += kin_row()
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0/sweep", "combiner sweep (frozen)")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0/tinysweep", "tiny sweep (frozen)")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0/multipool", "multipool (frozen)")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0_loraft", "combiner (LoRA-FT cache)")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0_loraft_multiscale", "combiner multi-scale (LoRA-FT)")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0/fine_only", "combiner fine-only basic")
    rows += combiner_dir_rows(BASE / "combiner/item_3_4/fold_0/fine_plus_coarse", "combiner fine+coarse basic")
    rows += tstats_rows(BASE / "temporal_stats/item_3_4/fold_0_frozen/velocity/temporal_encoder_results.json",
                        "tstats velocity partial (frozen)")
    rows += tstats_rows(BASE / "temporal_stats/item_3_4/fold_0_frozen/velocity_full/temporal_encoder_results.json",
                        "tstats velocity full (frozen)")
    rows += tstats_rows(BASE / "temporal_stats/item_3_4/fold_0_loraft/velocity_full/temporal_encoder_results.json",
                        "tstats velocity full (LoRA-FT)")

    # Sort by qwk descending
    rows.sort(key=lambda r: -r["qwk"])
    print(f"{'approach':<70} {'qwk':>8} {'acc':>8} {'mae':>8} {'n_val':>7}")
    print("-" * 110)
    for r in rows[:50]:
        print(f"{r['approach']:<70} {r['qwk']:>8.4f} {r['acc']:>8.4f} {r['mae']:>8.4f} {r['n_val']:>7}")


if __name__ == "__main__":
    main()
