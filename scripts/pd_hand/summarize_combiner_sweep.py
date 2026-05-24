#!/usr/bin/env python
"""Walk a combiner sweep directory and dump a sorted table of {qwk, acc, mae} per run.

Usage: python summarize_combiner_sweep.py <sweep_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", help="Directory containing one subdir per sweep config, each with result.json")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--rank-by", default="qwk", choices=["qwk", "acc", "mae"])
    args = p.parse_args()

    rows = []
    sweep_dir = Path(args.sweep_dir)
    for result_path in sorted(sweep_dir.glob("**/result.json")):
        try:
            r = json.loads(result_path.read_text())
        except Exception as exc:
            print(f"skip {result_path}: {exc}")
            continue
        best = r.get("best", {})
        rows.append({
            "tag": str(result_path.parent.relative_to(sweep_dir)),
            "qwk": best.get("qwk"),
            "acc": best.get("acc"),
            "mae": (best.get("metrics") or {}).get("mae"),
            "epoch": best.get("epoch"),
            "n_train": r.get("n_train"),
            "n_val": r.get("n_val"),
            "n_params_M": (r.get("n_params") or 0) / 1e6,
        })

    reverse = args.rank_by != "mae"
    rows.sort(key=lambda r: (r[args.rank_by] if r[args.rank_by] is not None else (-1e9 if reverse else 1e9)),
              reverse=reverse)

    print(f"{'tag':<45} {'qwk':>8} {'acc':>8} {'mae':>8} {'best_ep':>8} {'params_M':>9}")
    print("-" * 92)
    for row in rows[: args.top]:
        qwk = f"{row['qwk']:.4f}" if row["qwk"] is not None else "  --  "
        acc = f"{row['acc']:.4f}" if row["acc"] is not None else "  --  "
        mae = f"{row['mae']:.4f}" if row["mae"] is not None else "  --  "
        epoch = row["epoch"] if row["epoch"] is not None else "-"
        print(f"{row['tag']:<45} {qwk:>8} {acc:>8} {mae:>8} {epoch:>8} {row['n_params_M']:>9.2f}")

    if rows:
        best = rows[0]
        print()
        print(f"top by {args.rank_by}: {best['tag']}  qwk={best['qwk']:.4f}  acc={best['acc']:.4f}")
        print(f"n_train={best['n_train']}  n_val={best['n_val']}")


if __name__ == "__main__":
    main()
