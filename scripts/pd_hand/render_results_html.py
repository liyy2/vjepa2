#!/usr/bin/env python
"""Render an HTML dashboard summarizing all V-JEPA temporal-encoder sweeps.

Walks the vjepa2_temporal_encoder directory and renders:
  - Sweep cards (name, # configs, best single QWK/acc/MAE, top config)
  - A leaderboard of top-30 single configs across all sweeps
  - Best cross-sweep ensemble (greedy + top-K avg) using existing val_probs

Run after sweeps finish or while they're in progress (reads *partial* files too).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4")
    p.add_argument("--out", default="/gpfs/milgram/pi/scherzer/yl2428/vjepa2_sweeps.html")
    p.add_argument("--target-qwk", type=float, default=0.791)
    p.add_argument("--target-acc", type=float, default=0.682)
    return p.parse_args()


def load_sweep(sweep_dir: Path):
    """Return (name, items, best_overall_single) for a sweep dir.

    items: list of {qwk, val_acc, mae, config, val_probs?}
    """
    items = []
    files = []
    files.extend(sorted(sweep_dir.glob("shard*/temporal_encoder_results.json")))
    files.extend(sorted(sweep_dir.glob("shard*/temporal_encoder_partial_results.json")))
    files.extend(sorted(sweep_dir.glob("temporal_encoder_results.json")))
    files.extend(sorted(sweep_dir.glob("temporal_encoder_partial_results.json")))
    # de-dup: prefer final over partial
    seen_dirs = set()
    for f in files:
        if f.parent in seen_dirs and "partial" in f.name:
            continue
        seen_dirs.add(f.parent)
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("top_results", []):
            items.append({
                "qwk": r["qwk"], "val_acc": r["val_acc"], "mae": r["mae"],
                "config": r["config"],
                "val_probs": np.asarray(r["val_probs"], dtype=np.float32) if "val_probs" in r else None,
                "labels": np.asarray(r["labels"], dtype=np.int64) if "labels" in r else None,
                "epoch": r.get("epoch"),
            })
    if not items:
        return None
    items.sort(key=lambda x: x["qwk"], reverse=True)
    return items


def cross_ensemble(items, max_K=80):
    items = [it for it in items if it["val_probs"] is not None]
    if len(items) < 2:
        return None
    labels = items[0]["labels"]
    # Top-K average scan
    running = np.zeros_like(items[0]["val_probs"])
    best_topk = None
    for k, it in enumerate(items[:max_K], start=1):
        running += it["val_probs"]
        avg = running / k
        pred = avg.argmax(axis=1)
        qwk = float(cohen_kappa_score(labels, pred, weights="quadratic"))
        if best_topk is None or qwk > best_topk[0]:
            best_topk = (qwk, float(accuracy_score(labels, pred)), float(mean_absolute_error(labels, pred)), k)
    # Greedy
    pool = items[:max_K]
    chosen = []
    remaining = set(range(len(pool)))
    running = np.zeros_like(pool[0]["val_probs"])
    best_greedy = None
    history = []
    while remaining:
        cand = None
        for idx in remaining:
            trial = (running + pool[idx]["val_probs"]) / (len(chosen) + 1)
            pred = trial.argmax(axis=1)
            qwk = float(cohen_kappa_score(labels, pred, weights="quadratic"))
            if cand is None or qwk > cand[1]:
                cand = (idx, qwk, trial)
        idx, qwk, trial_avg = cand
        running += pool[idx]["val_probs"]
        chosen.append(idx)
        remaining.remove(idx)
        pred = trial_avg.argmax(axis=1)
        acc = float(accuracy_score(labels, pred))
        mae = float(mean_absolute_error(labels, pred))
        history.append((qwk, acc, mae, len(chosen)))
        if best_greedy is None or qwk > best_greedy[0]:
            best_greedy = (qwk, acc, mae, len(chosen))
        if len(history) > 12 and all(h[0] <= best_greedy[0] for h in history[-12:]):
            break
    return {"top_k": best_topk, "greedy": best_greedy}


def render_html(sweeps, all_items, target_qwk, target_acc):
    # rank all_items across sweeps
    all_items_sorted = sorted(all_items, key=lambda x: x["qwk"], reverse=True)
    ens = cross_ensemble(all_items)

    rows_sweeps = []
    for name, items in sweeps:
        if not items:
            continue
        best = items[0]
        c = best["config"]
        rows_sweeps.append(f"""
          <tr>
            <td>{name}</td>
            <td class="num">{len(items)}</td>
            <td class="num"><strong>{best['qwk']:.4f}</strong></td>
            <td class="num">{best['val_acc']:.4f}</td>
            <td class="num">{best['mae']:.4f}</td>
            <td class="mono small">{c.get('model_type','?')} loss={c.get('loss','?')} mixup={c.get('mixup_alpha','?')} drop={c.get('dropout','?')} hd={c.get('hidden_dim','?')} bins={c.get('temporal_bins','?')} fft={c.get('fft_bands','?')} fft_mode={c.get('fft_mode','?')} vel={c.get('use_velocity','?')} seed={c.get('seed','?')}</td>
          </tr>""")
    rows_top = []
    for r in all_items_sorted[:30]:
        c = r["config"]
        rows_top.append(f"""
          <tr>
            <td class="num">{r['qwk']:.4f}</td>
            <td class="num">{r['val_acc']:.4f}</td>
            <td class="num">{r['mae']:.4f}</td>
            <td class="mono small">{c.get('model_type','?')} loss={c.get('loss','?')} mixup={c.get('mixup_alpha','?')} cw={c.get('class_weight','?')} drop={c.get('dropout','?')} hd={c.get('hidden_dim','?')} bins={c.get('temporal_bins','?')} fft={c.get('fft_bands','?')} fft_mode={c.get('fft_mode','?')} vel={c.get('use_velocity','?')} seed={c.get('seed','?')}</td>
          </tr>""")

    ens_block = ""
    if ens:
        tk = ens["top_k"]; gd = ens["greedy"]
        ens_block = f"""
        <h2>Cross-sweep ensembles</h2>
        <div class="hero">
          <div class="stat"><div class="label">Best single (any sweep)</div>
            <div class="value">QWK {all_items_sorted[0]['qwk']:.4f}</div>
            <div class="sub">acc {all_items_sorted[0]['val_acc']:.4f} · MAE {all_items_sorted[0]['mae']:.4f}</div></div>
          <div class="stat"><div class="label">Top-K average (best K)</div>
            <div class="value">QWK {tk[0]:.4f}</div>
            <div class="sub">acc {tk[1]:.4f} · MAE {tk[2]:.4f} · K={tk[3]}</div></div>
          <div class="stat"><div class="label">Greedy forward ensemble</div>
            <div class="value">QWK {gd[0]:.4f}</div>
            <div class="sub">acc {gd[1]:.4f} · MAE {gd[2]:.4f} · K={gd[3]}</div></div>
          <div class="stat accent"><div class="label">Kinematic baseline target</div>
            <div class="value">QWK {target_qwk:.3f}</div>
            <div class="sub">acc {target_acc:.3f} — gap to greedy: {target_qwk - gd[0]:+.3f} QWK / {target_acc - gd[1]:+.3f} acc</div></div>
        </div>"""

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>V-JEPA temporal-encoder sweeps</title>
<style>
body{{background:#0f1115;color:#e7eaf0;font:14px/1.5 -apple-system,BlinkMacSystemFont,Inter,sans-serif;margin:0;padding:32px}}
.wrap{{max-width:1180px;margin:0 auto}}
h1{{font-size:24px;font-weight:600;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid #2a2f3a}}
.sub{{color:#9ba3b3;font-size:13px}}
.hero{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}}
.stat{{background:#171a21;border:1px solid #2a2f3a;border-radius:10px;padding:12px 14px}}
.stat .label{{color:#9ba3b3;font-size:11px;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px}}
.stat .value{{font-size:22px;font-weight:600}}
.stat.accent .value{{color:#a78bfa}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:6px 8px;border-bottom:1px solid #2a2f3a;text-align:left;vertical-align:top}}
th{{color:#9ba3b3;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:0.06em}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:"SF Mono",Menlo,monospace}} .small{{font-size:12px}}
.kinematic{{background:#1a2535}}
</style></head>
<body>
<div class="wrap">
<h1>V-JEPA temporal-encoder sweeps — item 3.4 fold 0</h1>
<div class="sub">Auto-rendered. Re-run scripts/pd_hand/render_results_html.py to refresh.</div>

<h2>Sweeps</h2>
<table>
  <thead><tr><th>Sweep</th><th class="num">configs</th><th class="num">best QWK</th><th class="num">best acc</th><th class="num">best MAE</th><th>top config</th></tr></thead>
  <tbody>{''.join(rows_sweeps)}
    <tr class="kinematic"><td><strong>kinematic baseline (ExtraTrees)</strong></td><td class="num">—</td><td class="num"><strong>{target_qwk:.4f}</strong></td><td class="num"><strong>{target_acc:.4f}</strong></td><td class="num">0.3333</td><td class="mono small">et_direct_s0_l2 (hand-engineered kinematic features)</td></tr>
  </tbody>
</table>
{ens_block}

<h2>Top 30 single configs across all sweeps</h2>
<table>
  <thead><tr><th class="num">QWK</th><th class="num">acc</th><th class="num">MAE</th><th>config</th></tr></thead>
  <tbody>{''.join(rows_top)}</tbody>
</table>
</div>
</body></html>"""
    return html


def main():
    args = parse_args()
    root = Path(args.root)
    sweeps = []
    all_items = []
    for sweep_dir in sorted(root.iterdir()):
        if not sweep_dir.is_dir():
            continue
        if "wandb" in sweep_dir.name:
            continue
        items = load_sweep(sweep_dir)
        if items is not None:
            sweeps.append((sweep_dir.name, items))
            all_items.extend(items)
    if not sweeps:
        print("no sweeps found")
        return
    html = render_html(sweeps, all_items, args.target_qwk, args.target_acc)
    Path(args.out).write_text(html)
    print(f"wrote {args.out} (sweeps={len(sweeps)}, total configs={len(all_items)})")


if __name__ == "__main__":
    main()
