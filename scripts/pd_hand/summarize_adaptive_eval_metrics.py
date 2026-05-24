#!/usr/bin/env python
"""Summarize adaptive item 3.4 eval metrics from config files.

Each eval config writes `metrics_latest.json` under:
  {folder}/{eval_name}/{tag}/metrics_latest.json

This script keeps the result table reproducible by deriving output locations
from the YAML configs instead of hand-copying paths from Slurm logs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML configs or glob patterns for adaptive eval configs.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--out-csv", type=Path, help="Optional CSV output path.")
    parser.add_argument("--out-md", type=Path, help="Optional Markdown table output path.")
    parser.add_argument(
        "--only-complete",
        action="store_true",
        help="Drop configs whose metrics_latest.json is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = _expand_configs(args.configs)
    rows = [_summarize_config(path) for path in config_paths]
    if args.only_complete:
        rows = [row for row in rows if row["status"] == "complete"]

    rows = sorted(rows, key=lambda row: (row["status"] != "complete", row["config"]))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else _fieldnames())
            writer.writeheader()
            writer.writerows(rows)
    markdown = _markdown_table(rows)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    else:
        print(markdown)


def _expand_configs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _summarize_config(config_path: Path) -> dict[str, Any]:
    row = {
        "config": str(config_path),
        "tag": "",
        "head_type": "",
        "checkpoint": "",
        "metrics_path": "",
        "best_path": "",
        "metric_source": "",
        "status": "missing_config",
        "epoch": "",
        "n": "",
        "accuracy": "",
        "qwk": "",
        "spearman": "",
        "mae": "",
        "coverage_rate": "",
        "temporal_span_rate": "",
        "best_classifier": "",
    }
    if not config_path.exists():
        return row

    cfg = yaml.safe_load(config_path.read_text()) or {}
    tag = str(cfg.get("tag") or "")
    eval_name = str(cfg.get("eval_name") or "")
    folder = Path(str(cfg.get("folder") or ""))
    run_dir = folder / eval_name / tag
    metrics_path = run_dir / "metrics_latest.json"
    best_path = run_dir / "best.pt"
    exp = cfg.get("experiment", {}) or {}
    classifier = exp.get("classifier", {}) or {}
    model_kwargs = cfg.get("model_kwargs", {}) or {}

    row.update(
        {
            "tag": tag,
            "head_type": str(classifier.get("head_type") or ""),
            "checkpoint": str(model_kwargs.get("checkpoint") or ""),
            "metrics_path": str(metrics_path),
            "best_path": str(best_path),
            "status": "missing_metrics",
        }
    )
    metrics = None
    if best_path.exists():
        metrics = _load_best_checkpoint_metrics(best_path)
        if metrics is not None:
            row["metric_source"] = "best.pt"
    if metrics is None and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        row["metric_source"] = "metrics_latest.json"
    if metrics is None:
        return row
    val = metrics.get("val", {}) or {}
    coverage = val.get("coverage", {}) or {}
    row.update(
        {
            "status": "complete",
            "epoch": metrics.get("epoch", ""),
            "n": val.get("n", val.get("raw_n", "")),
            "accuracy": _fmt(val.get("accuracy", val.get("val_acc", val.get("acc")))),
            "qwk": _fmt(val.get("quadratic_weighted_kappa", val.get("qwk"))),
            "spearman": _fmt(val.get("spearman")),
            "mae": _fmt(val.get("mae")),
            "coverage_rate": _fmt(coverage.get("coverage_rate")),
            "temporal_span_rate": _fmt(coverage.get("temporal_span_rate")),
            "best_classifier": val.get("best_classifier", ""),
        }
    )
    return row


def _fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def _load_best_checkpoint_metrics(path: Path) -> dict[str, Any] | None:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {
        "epoch": checkpoint.get("epoch", ""),
        "val": metrics,
        "selection_metric": checkpoint.get("selection_metric", ""),
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = [
        "config",
        "status",
        "epoch",
        "n",
        "accuracy",
        "qwk",
        "mae",
        "metric_source",
        "head_type",
        "tag",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            if field == "config":
                value = Path(str(value)).name
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _fieldnames() -> list[str]:
    return [
        "config",
        "tag",
        "head_type",
        "checkpoint",
        "metrics_path",
        "best_path",
        "metric_source",
        "status",
        "epoch",
        "n",
        "accuracy",
        "qwk",
        "spearman",
        "mae",
        "coverage_rate",
        "temporal_span_rate",
        "best_classifier",
    ]


if __name__ == "__main__":
    main()
