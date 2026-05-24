#!/usr/bin/env python
"""Walk the may19 run logs, extract per-epoch training trajectories, plot them,
and write an HTML report with embedded PNGs.

Sources:
  - Combiner stdout (sweep_*.out / tinysweep_*.out / run_*.out): one line per epoch with
        `epoch=NNN lr=... train_loss=... val_acc=... qwk=... mae=...`
  - LoRA-FT log_r0.csv: per-epoch CSV with train_acc + val_qwk (no train_loss)
  - Tstats output: only config-level summaries, no per-epoch loss

Output HTML: docs/results/may19_training_trajectories.html
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19"
)
EPOCH_RE = re.compile(
    r"epoch=(\d+) lr=([\d.eE+-]+) train_loss=([-\d.naNinf]+) val_acc=([\d.]+) qwk=([-\d.]+) mae=([\d.]+)"
)

# (run_dir_glob, label, color)
COMBINER_GROUPS = [
    (BASE / "combiner/item_3_4/fold_0/sweep/*/run_*.out", "frozen combiner sweep (21)", "#265e8a"),
    (BASE / "combiner/item_3_4/fold_0/tinysweep/*/run_*.out", "frozen tiny sweep (13)", "#1f7a47"),
    (BASE / "combiner/item_3_4/fold_0/multipool/*/run_*.out", "frozen multipool (11)", "#b88a00"),
    (BASE / "combiner/item_3_4/fold_0_loraft/*/run_*.out", "LoRA-FT combiner sweep (12)", "#c9430b"),
    (BASE / "combiner/item_3_4/fold_0_loraft_multiscale/*/run_*.out", "LoRA-FT multi-scale (12)", "#b22222"),
    (BASE / "combiner/item_3_4/fold_0_loraft_distill/*/run_*.out", "LoRA-FT distill (15)", "#5b4b8a"),
    (BASE / "combiner/item_3_4/fold_0_frozen_plus_loraft/*/run_*.out", "frozen+LoRA-FT 2-stream (15)", "#a04550"),
]


def parse_combiner_run(path: Path) -> list[dict]:
    """Return list of {epoch, lr, train_loss, val_acc, qwk, mae}."""
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            rows.append(dict(
                epoch=int(m.group(1)),
                lr=float(m.group(2)),
                train_loss=float(m.group(3)) if m.group(3) not in ("nan",) else float("nan"),
                val_acc=float(m.group(4)),
                qwk=float(m.group(5)),
                mae=float(m.group(6)),
            ))
    return rows


def parse_combiner_runs_dir(glob_pattern: Path) -> dict[str, list[dict]]:
    """Group: when stdout is one file per sweep (sweep_*.out, run_*.out at sweep root),
    parse and split by config section (`=== tag ===` markers)."""
    out: dict[str, list[dict]] = {}
    # Many sweeps put one out file at sweep root with all configs concatenated. Treat that case.
    paths = list(Path("/").glob(str(glob_pattern).lstrip("/")))
    if not paths:
        # Try also reading the sweep stdout at the parent level
        parent_pattern = str(glob_pattern.parent.parent / glob_pattern.name)
        paths = [Path(p) for p in Path("/").glob(parent_pattern.lstrip("/"))]
    for p in paths:
        text = p.read_text(errors="ignore")
        # Split by "=== TAG ===" markers
        sections = re.split(r"^=== (.+?) ===\s*$", text, flags=re.MULTILINE)
        if len(sections) > 1:
            # sections[0] is preamble, then alternating tag/content
            for i in range(1, len(sections), 2):
                tag = sections[i].strip()
                content = sections[i + 1]
                # Walk content, extract epoch lines
                rows = []
                for line in content.splitlines():
                    m = EPOCH_RE.search(line)
                    if m:
                        rows.append(dict(
                            epoch=int(m.group(1)),
                            train_loss=float(m.group(3)) if m.group(3) not in ("nan",) else float("nan"),
                            qwk=float(m.group(5)),
                            val_acc=float(m.group(4)),
                        ))
                if rows:
                    out[f"{p.parent.name}/{tag}"] = rows
        else:
            rows = parse_combiner_run(p)
            if rows:
                out[p.parent.name] = rows
    return out


def parse_loraft_csv() -> list[dict]:
    """Parse LoRA-FT log_r0.csv (has train_acc + val_qwk; no train_loss)."""
    csv = (BASE / "vjepa2_evals/item_3_4/fold_0_supervised_lora_ft/video_classification_frozen/"
                  "pd-hand-may19-item-3_4-fold-0-supervised-lora-ft/log_r0.csv")
    if not csv.exists():
        return []
    rows = []
    import csv as csvmod
    with csv.open() as f:
        for r in csvmod.DictReader(f):
            rows.append(dict(
                epoch=int(r["epoch"]),
                train_acc=float(r["train_acc"]),
                val_acc=float(r["val_acc"]),
                val_qwk=float(r["val_qwk"]),
                val_spearman=float(r["val_spearman"]),
                val_mae=float(r["val_mae"]),
            ))
    return rows


def make_combiner_panel(group_label: str, glob_path: Path, color: str, fig_h=4) -> str:
    """Make a 2-panel figure: train_loss curves + val_qwk curves, all runs in this group."""
    out = parse_combiner_runs_dir(glob_path)
    if not out:
        return ""
    runs = list(out.items())
    fig, ax = plt.subplots(1, 2, figsize=(11, fig_h), constrained_layout=True)
    for name, rows in runs:
        e = [r["epoch"] for r in rows]
        tl = [r["train_loss"] for r in rows]
        q = [r["qwk"] for r in rows]
        ax[0].plot(e, tl, color=color, alpha=0.35, linewidth=1.0)
        ax[1].plot(e, q, color=color, alpha=0.35, linewidth=1.0)
    # Highlight the best run by max QWK
    best_name = max(runs, key=lambda r: max((row["qwk"] for row in r[1]), default=0))[0]
    best_rows = dict(runs)[best_name]
    ax[0].plot([r["epoch"] for r in best_rows], [r["train_loss"] for r in best_rows],
               color=color, linewidth=2.2, label=f"best: {best_name}")
    ax[1].plot([r["epoch"] for r in best_rows], [r["qwk"] for r in best_rows],
               color=color, linewidth=2.2)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train_loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val QWK")
    ax[0].set_title(f"{group_label} — train_loss (n={len(runs)} runs)")
    ax[1].set_title("val QWK")
    ax[1].axhline(0.7277, color="#1f7a47", linestyle="--", linewidth=1, alpha=0.7, label="kinematic 0.728")
    ax[1].axhline(0.5328, color="#b88a00", linestyle=":", linewidth=1, alpha=0.7, label="best frozen ensemble 0.533")
    ax[1].legend(loc="lower right", fontsize=8)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(alpha=0.2); ax[1].grid(alpha=0.2)
    ax[1].set_ylim(-0.1, 0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_loraft_panel() -> str:
    rows = parse_loraft_csv()
    if not rows:
        return ""
    e = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    ax[0].plot(e, [r["train_acc"] for r in rows], color="#265e8a", label="train_acc", linewidth=2)
    ax[0].plot(e, [r["val_acc"] for r in rows], color="#c9430b", label="val_acc", linewidth=2)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy (%)"); ax[0].grid(alpha=0.2)
    ax[0].set_title("LoRA-FT (softmax, unweighted)\nacc")
    ax[0].legend()
    ax[1].plot(e, [r["val_qwk"] for r in rows], color="#1f7a47", linewidth=2, label="val QWK")
    ax[1].plot(e, [r["val_spearman"] for r in rows], color="#5b4b8a", linewidth=2, label="val Spearman")
    ax[1].axhline(0.7277, color="#1f7a47", linestyle="--", linewidth=1, alpha=0.7, label="kinematic 0.728")
    ax[1].axhline(0.5328, color="#b88a00", linestyle=":", linewidth=1, alpha=0.7, label="frozen ensemble 0.533")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metric"); ax[1].grid(alpha=0.2)
    ax[1].set_title("LoRA-FT\nval QWK / Spearman")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_ylim(-0.05, 0.8)
    ax[2].plot(e, [r["val_mae"] for r in rows], color="#b22222", linewidth=2, label="val MAE")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("MAE"); ax[2].grid(alpha=0.2)
    ax[2].set_title("LoRA-FT\nval MAE")
    ax[2].legend()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    out_path = Path("/gpfs/milgram/pi/scherzer/yl2428/vjepa2/docs/results/may19_training_trajectories.html")

    # Build all panels
    print("Parsing LoRA-FT...")
    loraft_png = make_loraft_panel()

    combiner_imgs = []
    for glob_path, label, color in COMBINER_GROUPS:
        print(f"Parsing {label}...")
        b64 = make_combiner_panel(label, glob_path, color)
        if b64:
            combiner_imgs.append((label, b64))

    html_parts = ["""<!doctype html>
<html lang=en>
<head><meta charset=utf-8><title>may19 training trajectories</title>
<style>
body { background: #fffdf8; color: #1a1a1a; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 36px auto; padding: 0 26px; line-height: 1.5; }
h1 { font-size: 28px; margin-bottom: 4px; }
h2 { font-size: 19px; margin-top: 36px; padding-bottom: 4px; border-bottom: 1px solid #d8d4cc; }
.fig { margin: 16px 0; padding: 8px; background: #faf7f0; border: 1px solid #d8d4cc; border-radius: 4px; }
.fig img { max-width: 100%; display: block; }
.callout { background: #f3efe5; border-left: 3px solid #c9430b; padding: 10px 16px; margin: 14px 0; font-size: 14px; }
code { font-family: "JetBrains Mono", Menlo, monospace; font-size: 12.5px; background: #faf7f0; padding: 1px 4px; border-radius: 3px; }
</style></head>
<body>
<h1>may19 training trajectories</h1>
<p>Per-epoch <code>train_loss</code> + <code>val QWK</code> curves for every combiner sweep,
plus the per-epoch CSV log for the LoRA-FT supervised probe. Kinematic baseline (LightGBM)
has no per-epoch concept and the tstats grid only logs config-level summaries, so they
don't appear here.</p>

<div class="callout">
<strong>What to look for.</strong>
The combiner runs overfit aggressively: train_loss → 0.0 within 30–60 epochs while val QWK
plateaus around 0.4–0.5. Larger models overfit fastest. The LoRA-FT probe's val Spearman
climbs monotonically (0.46 → 0.53) but val QWK is noisy because the softmax argmax bounces
between collapsed and recovering — a sign that the encoder learns a usable ordering but the
categorical head can't decide thresholds.
</div>
"""]

    if loraft_png:
        html_parts.append('<h2>Supervised LoRA-FT (softmax, unweighted) — job 28870558</h2>')
        html_parts.append(f'<div class="fig"><img src="data:image/png;base64,{loraft_png}"></div>')

    for label, b64 in combiner_imgs:
        html_parts.append(f'<h2>{label}</h2>')
        html_parts.append(f'<div class="fig"><img src="data:image/png;base64,{b64}"></div>')

    html_parts.append("""
<h2>Notes on omissions</h2>
<ul>
  <li><strong>Kinematic LightGBM</strong>: gradient boosting has per-iteration <em>tree-add</em>
      training, but no per-epoch loss in the sense used here. The
      <code>results_stride2.json</code> records only the best-iteration metric.</li>
  <li><strong>Tstats velocity grid</strong>: <code>train_temporal_encoder.py</code> logs only
      <code>new_best</code> lines (config-level summaries) to stdout; it does not write
      per-epoch loss curves. To get those, the script would need a per-config CSV writer
      added to <code>fit_one</code>.</li>
  <li><strong>V-JEPA cache jobs</strong>: pure inference, no training.</li>
</ul>
</body></html>
""")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html_parts))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
