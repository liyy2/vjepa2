#!/usr/bin/env python
"""Publication figure for the CLEAN hybrid result.

kin+dx vs kin+dx+V-JEPA-OOF-probs on may19 fold 0, 10 ExtraTrees seeds.
This is the leak-free version (no item 3.5 UPDRS labels).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

SKILL = "/home/yl2428/.claude/skills/cns-scientific-plots"
sys.path.insert(0, f"{SKILL}/scripts")
from cns_plot_style import (
    apply_cns_style,
    add_panel_label,
    get_palette,
    save_publication_figure,
)


def main():
    DATA = Path("/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_0/oof_stack/multiseed_oof_probs.json")
    d = json.loads(DATA.read_text())

    base_per = d["kin_plus_dx"]["per_seed"]
    hyb_per = d["kin_plus_dx_plus_vjepa_all"]["per_seed"]
    seeds = [r["seed"] for r in base_per]
    base_q = np.array([r["qwk"] for r in base_per])
    hyb_q = np.array([r["qwk"] for r in hyb_per])
    delta = hyb_q - base_q

    stat = wilcoxon(hyb_q, base_q, alternative="greater")
    print(f"baseline:  {base_q.mean():.4f} ± {base_q.std():.4f}")
    print(f"hybrid:    {hyb_q.mean():.4f} ± {hyb_q.std():.4f}")
    print(f"Δ:         +{delta.mean():.4f}  median={np.median(delta):+.4f}  ({(delta>0).sum()}/{len(delta)} wins)")
    print(f"Wilcoxon p={stat.pvalue:.4f}")

    apply_cns_style(context="panel")
    palette = get_palette("editorial")
    KIN_COLOR = palette[0]
    HYB_COLOR = palette[1]
    WIN_COLOR = "#4F5561"
    LOSS_COLOR = palette[2]

    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.45,
                          left=0.09, right=0.97, bottom=0.15, top=0.83)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # =================== Panel a: paired strip with connecting lines ===================
    x_kin, x_hyb = 0.0, 1.0
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.02, size=len(seeds))

    for i, (k, h) in enumerate(zip(base_q, hyb_q)):
        color = LOSS_COLOR if h < k else WIN_COLOR
        alpha = 0.5 if h >= k else 0.9
        ax_a.plot([x_kin + jitter[i], x_hyb + jitter[i]], [k, h],
                  color=color, linewidth=0.9, alpha=alpha, zorder=1)
    ax_a.scatter([x_kin + j for j in jitter], base_q,
                 s=42, color=KIN_COLOR, edgecolor="white", linewidth=0.8,
                 zorder=3, label="kinematic + dx")
    ax_a.scatter([x_hyb + j for j in jitter], hyb_q,
                 s=42, color=HYB_COLOR, edgecolor="white", linewidth=0.8,
                 zorder=3, label="+ V-JEPA OOF probs")
    ax_a.scatter([x_kin], [base_q.mean()], s=120, color=KIN_COLOR,
                 marker="D", edgecolor="white", linewidth=1.4, zorder=5)
    ax_a.scatter([x_hyb], [hyb_q.mean()], s=120, color=HYB_COLOR,
                 marker="D", edgecolor="white", linewidth=1.4, zorder=5)

    ax_a.set_xticks([x_kin, x_hyb])
    ax_a.set_xticklabels([
        "kinematic + dx\n(no item 3.5)",
        "+ V-JEPA OOF probs\n(LoRA-FT adaptive)"
    ], fontsize=9)
    ax_a.set_ylabel("Quadratic-weighted κ", fontsize=10)
    ax_a.set_xlim(-0.55, 1.55)
    ax_a.set_ylim(0.50, 0.65)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.axhline(base_q.mean(), color=KIN_COLOR, linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
    ax_a.axhline(hyb_q.mean(), color=HYB_COLOR, linestyle=":", linewidth=0.7, alpha=0.5, zorder=0)
    add_panel_label(ax_a, "a", lowercase=True)

    ax_a.text(x_kin - 0.18, base_q.mean(), f"{base_q.mean():.3f}",
              ha="right", va="center", color=KIN_COLOR, fontsize=9, fontweight="bold")
    ax_a.text(x_hyb + 0.18, hyb_q.mean(), f"{hyb_q.mean():.3f}",
              ha="left", va="center", color=HYB_COLOR, fontsize=9, fontweight="bold")

    n_win = int((delta > 0).sum())
    p_str = f"p = {stat.pvalue:.3f}" if stat.pvalue >= 0.001 else f"p = {stat.pvalue:.0e}"
    ax_a.text(0.97, 0.97,
              f"Δ = +{delta.mean():.3f} ± {delta.std():.3f}\n{n_win}/{len(delta)} seeds, {p_str}",
              transform=ax_a.transAxes,
              ha="right", va="top", fontsize=8.5,
              bbox=dict(boxstyle="round,pad=0.36", facecolor="#FAF7F0",
                        edgecolor="#D8D4CC", linewidth=0.6))

    # =================== Panel b: bar with mean ± SD ===================
    width = 0.5
    xs = [0.0, 1.0]
    means = [base_q.mean(), hyb_q.mean()]
    sds = [base_q.std(), hyb_q.std()]
    colors = [KIN_COLOR, HYB_COLOR]
    bars = ax_b.bar(xs, means, width=width, color=colors, edgecolor="white",
                     linewidth=1.0, zorder=2)
    ax_b.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#1F1F1F", capsize=4,
                  linewidth=1.0, zorder=3)
    for i, (k, h) in enumerate(zip(base_q, hyb_q)):
        ax_b.scatter(xs[0] + jitter[i] * 0.5, k, s=14, color=KIN_COLOR,
                     edgecolor="white", linewidth=0.6, alpha=0.85, zorder=4)
        ax_b.scatter(xs[1] + jitter[i] * 0.5, h, s=14, color=HYB_COLOR,
                     edgecolor="white", linewidth=0.6, alpha=0.85, zorder=4)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(["kin + dx", "+ V-JEPA OOF"], fontsize=9)
    ax_b.set_ylabel("Quadratic-weighted κ", fontsize=10)
    ax_b.set_ylim(0, 0.72)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    for x, m, s in zip(xs, means, sds):
        ax_b.text(x, m + s + 0.012, f"{m:.3f}", ha="center", va="bottom",
                  fontsize=8.5, fontweight="bold", color="#1F1F1F")
    y_bracket = max(means) + max(sds) + 0.05
    ax_b.plot([xs[0], xs[0], xs[1], xs[1]],
              [y_bracket - 0.015, y_bracket, y_bracket, y_bracket - 0.015],
              color="#1F1F1F", linewidth=0.8)
    p_label = "***" if stat.pvalue < 0.001 else ("**" if stat.pvalue < 0.01 else ("*" if stat.pvalue < 0.05 else "n.s."))
    ax_b.text((xs[0] + xs[1]) / 2, y_bracket + 0.005, p_label,
              ha="center", va="bottom", fontsize=10, fontweight="bold")
    add_panel_label(ax_b, "b", lowercase=True)

    fig.text(0.09, 0.93, "V-JEPA OOF combiner predictions improve the clean kinematic baseline",
             ha="left", va="bottom", fontsize=12, fontweight="bold")
    fig.text(0.09, 0.895,
             "MDS-UPDRS item 3.4, may19 fold 0 (n=107 val); no item 3.5 label leak; 10 ExtraTrees seeds (n_est=1000)",
             ha="left", va="bottom", fontsize=8.5, color="#4F5561")

    out = Path("/gpfs/milgram/pi/scherzer/yl2428/vjepa2/docs/results/figure_hybrid_clean_qwk")
    save_publication_figure(fig, out)
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
