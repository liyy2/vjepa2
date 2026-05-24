#!/usr/bin/env python
"""Publication figure: kin vs kin + V-JEPA QWK on may19 fold 0, 10 seeds.

Panel A: paired strip with connecting lines (one line per seed)
Panel B: bar with mean ± SD

Writes figure_hybrid_qwk.pdf (vector) + .png (preview).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

# Add the cns-plot-style helper
SKILL = "/home/yl2428/.claude/skills/cns-scientific-plots"
sys.path.insert(0, f"{SKILL}/scripts")
from cns_plot_style import (
    apply_cns_style,
    create_panel_figure,
    add_panel_label,
    get_palette,
    save_publication_figure,
)


def main():
    DATA = Path("/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_0/multiseed_coarse_step2_n1000.json")
    d = json.loads(DATA.read_text())

    kin = d["kin_plus_meta"]["per_seed"]
    hyb = d["kin_plus_meta_plus_vjepa"]["per_seed"]
    seeds = [r["seed"] for r in kin]
    kin_q = np.array([r["qwk"] for r in kin])
    hyb_q = np.array([r["qwk"] for r in hyb])
    delta = hyb_q - kin_q

    # Wilcoxon signed-rank test for paired difference
    stat = wilcoxon(hyb_q, kin_q, alternative="greater")
    print(f"Wilcoxon: stat={stat.statistic:.2f}, p_one_sided={stat.pvalue:.4f}")
    print(f"kin mean ± std:    {kin_q.mean():.4f} ± {kin_q.std():.4f}")
    print(f"hybrid mean ± std: {hyb_q.mean():.4f} ± {hyb_q.std():.4f}")
    print(f"mean lift:         +{delta.mean():.4f} (n={len(delta)}, {(delta>0).sum()}/{len(delta)} positive)")

    apply_cns_style(context="panel")
    palette = get_palette("editorial")
    KIN_COLOR = palette[0]      # slate blue
    HYB_COLOR = palette[1]      # dusty coral
    WIN_COLOR = "#4F5561"       # dark gray for connecting lines (wins)
    LOSS_COLOR = palette[2]     # sage (the one loss)

    fig, axes = create_panel_figure(nrows=1, ncols=2, width="single",
                                     panel_aspect=1.0, squeeze=False)
    # create_panel_figure with width="single" returns ~3.5 inches; we want a wider double for two panels
    # Use a custom figure instead
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.45,
                          left=0.09, right=0.97, bottom=0.15, top=0.83)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # =================== Panel A: paired strip with connecting lines ===================
    x_kin, x_hyb = 0.0, 1.0
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.02, size=len(seeds))

    for i, (k, h) in enumerate(zip(kin_q, hyb_q)):
        # Connecting line color: gray for V-JEPA-win, sage for V-JEPA-loss
        color = LOSS_COLOR if h < k else WIN_COLOR
        alpha = 0.55 if h >= k else 0.85
        ax_a.plot([x_kin + jitter[i], x_hyb + jitter[i]], [k, h],
                  color=color, linewidth=0.9, alpha=alpha, zorder=1)
    # Markers on top
    ax_a.scatter([x_kin + j for j in jitter], kin_q,
                 s=42, color=KIN_COLOR, edgecolor="white", linewidth=0.8,
                 zorder=3, label="kinematic")
    ax_a.scatter([x_hyb + j for j in jitter], hyb_q,
                 s=42, color=HYB_COLOR, edgecolor="white", linewidth=0.8,
                 zorder=3, label="kin + V-JEPA")

    # Mean markers (larger, opaque, on top)
    ax_a.scatter([x_kin], [kin_q.mean()], s=120, color=KIN_COLOR,
                 marker="D", edgecolor="white", linewidth=1.4, zorder=5)
    ax_a.scatter([x_hyb], [hyb_q.mean()], s=120, color=HYB_COLOR,
                 marker="D", edgecolor="white", linewidth=1.4, zorder=5)

    ax_a.set_xticks([x_kin, x_hyb])
    ax_a.set_xticklabels(["kinematic\n(old_118)", "kin + V-JEPA\n(coarse_step2)"], fontsize=9)
    ax_a.set_ylabel("Quadratic-weighted κ", fontsize=10)
    ax_a.set_xlim(-0.55, 1.55)
    ax_a.set_ylim(0.685, 0.80)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Subtle dashed line for kin mean
    ax_a.axhline(kin_q.mean(), color=KIN_COLOR, linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
    ax_a.axhline(hyb_q.mean(), color=HYB_COLOR, linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)

    add_panel_label(ax_a, "a", lowercase=True)
    # Direct labels for the means (placed slightly above to avoid overlap with markers)
    ax_a.text(x_kin - 0.18, kin_q.mean(), f"{kin_q.mean():.3f}",
              ha="right", va="center", color=KIN_COLOR, fontsize=9, fontweight="bold")
    ax_a.text(x_hyb + 0.18, hyb_q.mean(), f"{hyb_q.mean():.3f}",
              ha="left", va="center", color=HYB_COLOR, fontsize=9, fontweight="bold")

    # Annotate the lift, top-right corner (out of the data cloud)
    n_win = int((delta > 0).sum())
    p_str = f"p = {stat.pvalue:.3f}" if stat.pvalue >= 0.001 else f"p = {stat.pvalue:.0e}"
    ax_a.text(0.97, 0.97,
              f"Δ = +{delta.mean():.3f} ± {delta.std():.3f}\n{n_win}/{len(delta)} seeds, {p_str}",
              transform=ax_a.transAxes,
              ha="right", va="top", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.36", facecolor="#FAF7F0",
                        edgecolor="#D8D4CC", linewidth=0.6))

    # =================== Panel B: bar with mean ± SD ===================
    width = 0.5
    xs = [0.0, 1.0]
    means = [kin_q.mean(), hyb_q.mean()]
    sds = [kin_q.std(), hyb_q.std()]
    colors = [KIN_COLOR, HYB_COLOR]
    bars = ax_b.bar(xs, means, width=width, color=colors, edgecolor="white",
                     linewidth=1.0, zorder=2)
    ax_b.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#1F1F1F", capsize=4,
                  linewidth=1.0, zorder=3)
    # Overlay individual seed points
    for i, (k, h) in enumerate(zip(kin_q, hyb_q)):
        ax_b.scatter(xs[0] + jitter[i] * 0.5, k, s=14, color=KIN_COLOR,
                     edgecolor="white", linewidth=0.6, alpha=0.85, zorder=4)
        ax_b.scatter(xs[1] + jitter[i] * 0.5, h, s=14, color=HYB_COLOR,
                     edgecolor="white", linewidth=0.6, alpha=0.85, zorder=4)

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(["kinematic", "kin + V-JEPA"], fontsize=9)
    ax_b.set_ylabel("Quadratic-weighted κ", fontsize=10)
    ax_b.set_ylim(0, 0.85)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Value labels on bars
    for x, m, s in zip(xs, means, sds):
        ax_b.text(x, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color="#1F1F1F")

    # Significance annotation
    y_bracket = max(means) + max(sds) + 0.05
    ax_b.plot([xs[0], xs[0], xs[1], xs[1]],
              [y_bracket - 0.015, y_bracket, y_bracket, y_bracket - 0.015],
              color="#1F1F1F", linewidth=0.8)
    p_label = "***" if stat.pvalue < 0.001 else ("**" if stat.pvalue < 0.01 else ("*" if stat.pvalue < 0.05 else "n.s."))
    ax_b.text((xs[0] + xs[1]) / 2, y_bracket + 0.005, p_label,
              ha="center", va="bottom", fontsize=10, fontweight="bold")

    add_panel_label(ax_b, "b", lowercase=True)

    # Header title
    fig.text(0.09, 0.93, "V-JEPA features add to the kinematic baseline",
             ha="left", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.09, 0.895,
             f"MDS-UPDRS item 3.4 (finger tapping), may19 fold 0 (n=107 val); 10 ExtraTrees seeds, n_est=1000, balanced",
             ha="left", va="bottom", fontsize=8.5, color="#4F5561")

    out = Path("/gpfs/milgram/pi/scherzer/yl2428/vjepa2/docs/results/figure_hybrid_qwk")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, out)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
