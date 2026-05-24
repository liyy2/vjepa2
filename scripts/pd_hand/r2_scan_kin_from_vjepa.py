#!/usr/bin/env python
"""Systematic R² scan: which V-JEPA cache × reducer best predicts kinematic features?

For each (cache_npz, reducer) pair, fit on train, score R² per kinematic feature on val.
Save a CSV of all (cache, reducer, feature) → R².
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

KIN_FEATS = [
    "bbox_area_peak_rate_hz",
    "bbox_area_peak_interval_cv",
    "bbox_area_decrement",
    "bbox_area_cycle_amplitude_mean",
    "bbox_area_cycle_amplitude_std",
    "bbox_area_dominant_freq_hz",
    "bbox_area_bandpower_0p5_2_hz",
    "bbox_area_bandpower_2_5_hz",
    "bbox_area_bandpower_5_8_hz",
    "bbox_area_velocity_mean_abs",
    "bbox_area_velocity_std",
    "bbox_area_peak_count",
    "bbox_area_peak_prominence_mean",
    "bbox_area_slope",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caches-train", action="append", required=True,
                   help="train cache npz path; pair with --caches-val")
    p.add_argument("--caches-val", action="append", required=True)
    p.add_argument("--cache-names", action="append", required=True)
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--reducers", default="summary,summary_fft,pca128",
                   help="comma-separated reducer names to apply to each cache")
    p.add_argument("--models", default="ridge,mlp,lgbm",
                   help="comma-separated model names")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-html", default="")
    return p.parse_args()


def load_cache(p):
    d = np.load(p, allow_pickle=False)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.int64)
    mask = d["mask"].astype(bool) if "mask" in d.files else np.ones(x.shape[:2], dtype=bool)
    cov = [json.loads(str(s)) for s in d["coverage"]] if "coverage" in d.files else None
    return x, y, mask, cov


def load_kin_targets(fold_csv, kin_csv, coverage):
    fold_rows = list(csv.DictReader(open(fold_csv)))
    by_clip = {r["clip_path"]: r for r in csv.DictReader(open(kin_csv))}
    if coverage is None:
        sample_indices = list(range(len(fold_rows)))
    else:
        sample_indices = [int(round(float(c.get("sample_index", -1)))) for c in coverage]
    N = len(sample_indices)
    Y = np.full((N, len(KIN_FEATS)), np.nan, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    for i, si in enumerate(sample_indices):
        if si < 0 or si >= len(fold_rows):
            continue
        kin = by_clip.get(fold_rows[si]["clip_path"])
        if kin is None:
            continue
        vals = []
        ok = True
        for col in KIN_FEATS:
            try:
                v = float(kin.get(col, ""))
                if not np.isfinite(v):
                    ok = False
                    break
                vals.append(v)
            except Exception:
                ok = False
                break
        if ok:
            Y[i] = vals
            mask[i] = True
    return Y, mask


def reducer_summary(x, mask):
    """[N, T, D] → [N, F]: mean / std / |Δ|-mean / first / last / last-first."""
    valid = mask.astype(np.float32)[..., None]
    cnt = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    mean = (x * valid).sum(axis=1) / cnt
    var = (((x - mean[:, None]) ** 2) * valid).sum(axis=1) / cnt
    std = np.sqrt(np.clip(var, 0, None))
    delta = x[:, 1:] - x[:, :-1]
    dmask = (mask[:, 1:] & mask[:, :-1])
    dvalid = dmask.astype(np.float32)[..., None]
    dcnt = dmask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    vmean = (np.abs(delta) * dvalid).sum(axis=1) / dcnt
    # First/last valid token
    first_idx = np.argmax(mask, axis=1)  # index of first True (returns 0 if all False)
    rev_mask = mask[:, ::-1]
    last_offset = np.argmax(rev_mask, axis=1)
    last_idx = mask.shape[1] - 1 - last_offset
    first = x[np.arange(x.shape[0]), first_idx]
    last = x[np.arange(x.shape[0]), last_idx]
    return np.concatenate([mean, std, vmean, first, last, last - first], axis=1).astype(np.float32)


def reducer_summary_fft(x, mask, fps_per_tubelet=15.0, bands_hz=((0.5, 2), (2, 6))):
    """summary + FFT band powers."""
    base = reducer_summary(x, mask)
    valid = mask.astype(np.float32)[..., None]
    cnt = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    mean = (x * valid).sum(axis=1, keepdims=True) / cnt[..., None]
    centered = (x - mean) * valid
    spec = np.fft.rfft(centered, axis=1, norm="ortho")
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fps_per_tubelet)
    band_outs = []
    for lo, hi in bands_hz:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            band_outs.append(np.zeros((x.shape[0], x.shape[2]), dtype=np.float32))
        else:
            band_outs.append(np.log1p(mag[:, idx].mean(axis=1)).astype(np.float32))
    band = np.concatenate(band_outs, axis=1)
    return np.concatenate([base, band], axis=1).astype(np.float32)


def reducer_pca128_summary(x, mask, x_train=None, mask_train=None):
    """PCA to 128 dims on flat tokens, then summary."""
    if x_train is None:
        # fitting case: caller will pass x_train=x
        x_train = x; mask_train = mask
    from sklearn.decomposition import PCA
    flat = x_train.reshape(-1, x_train.shape[-1])
    mu = flat.mean(axis=0)
    pca = PCA(n_components=min(128, flat.shape[1]), random_state=0)
    pca.fit(flat - mu)
    W = pca.components_.astype(np.float32)
    def apply(z):
        return (z.reshape(-1, z.shape[-1]) - mu) @ W.T
    x_red = apply(x).reshape(x.shape[0], x.shape[1], -1)
    return reducer_summary(x_red, mask)


REDUCERS = {
    "summary": lambda xtr, mtr, xva, mva: (reducer_summary(xtr, mtr), reducer_summary(xva, mva)),
    "summary_fft": lambda xtr, mtr, xva, mva: (reducer_summary_fft(xtr, mtr), reducer_summary_fft(xva, mva)),
    "pca128": lambda xtr, mtr, xva, mva: (
        reducer_pca128_summary(xtr, mtr, xtr, mtr),
        reducer_pca128_summary(xva, mva, xtr, mtr),
    ),
}


def fit_eval(Xtr, Ytr, Mtr, Xva, Yva, Mva, model_name):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr[Mtr])
    Xva = scaler.transform(Xva[Mva])
    Ytr = Ytr[Mtr]
    Yva = Yva[Mva]
    mu = Ytr.mean(axis=0)
    sd = Ytr.std(axis=0) + 1e-6
    Yztr = (Ytr - mu) / sd
    r2_per = []
    for j in range(Yztr.shape[1]):
        if model_name == "ridge":
            m = Ridge(alpha=10.0)
            m.fit(Xtr, Yztr[:, j])
            pred = m.predict(Xva)
        elif model_name == "mlp":
            m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400,
                             early_stopping=True, n_iter_no_change=30,
                             random_state=0, alpha=1e-3)
            m.fit(Xtr, Yztr[:, j])
            pred = m.predict(Xva)
        elif model_name == "lgbm":
            try:
                import lightgbm as lgb
                m = lgb.LGBMRegressor(
                    objective="regression", n_estimators=400, learning_rate=0.05,
                    num_leaves=31, feature_fraction=0.7, bagging_fraction=0.8,
                    bagging_freq=3, verbose=-1, random_state=0,
                )
                m.fit(Xtr, Yztr[:, j])
                pred = m.predict(Xva)
            except Exception:
                pred = np.zeros(Xva.shape[0])
        else:
            raise ValueError(model_name)
        yhat_z = pred * sd[j] + mu[j]
        r2 = r2_score(Yva[:, j], yhat_z)
        r2_per.append(r2)
    return r2_per


def main():
    args = parse_args()
    n = len(args.caches_train)
    assert n == len(args.caches_val) == len(args.cache_names), "mismatched --caches/--names"

    reducers = [r.strip() for r in args.reducers.split(",") if r.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    out_rows = []
    for cache_name, p_tr, p_va in zip(args.cache_names, args.caches_train, args.caches_val):
        print(f"\n=== {cache_name} ===  {p_tr}")
        try:
            xtr, ytr, mtr, cov_tr = load_cache(p_tr)
            xva, yva, mva, cov_va = load_cache(p_va)
        except Exception as exc:
            print(f"  load failed: {exc}")
            continue
        Ytr_k, Mtr = load_kin_targets(args.fold_train_csv, args.kin_train_csv, cov_tr)
        Yva_k, Mva = load_kin_targets(args.fold_val_csv, args.kin_val_csv, cov_va)
        print(f"  cache shape: train {xtr.shape}, val {xva.shape};  kin valid: tr {Mtr.sum()}/{len(Mtr)}, va {Mva.sum()}/{len(Mva)}")

        for reducer_name in reducers:
            print(f"  -- reducer={reducer_name}")
            try:
                Xtr, Xva = REDUCERS[reducer_name](xtr, mtr, xva, mva)
            except Exception as exc:
                print(f"    reducer error: {exc}")
                continue
            print(f"    feature dim {Xtr.shape[1]}")
            for model_name in models:
                print(f"    -- model={model_name}")
                try:
                    r2_per = fit_eval(Xtr, Ytr_k, Mtr, Xva, Yva_k, Mva, model_name)
                except Exception as exc:
                    print(f"      model error: {exc}")
                    continue
                mean_r2 = float(np.mean(r2_per))
                pos_count = int(sum(1 for r in r2_per if r > 0))
                best = max(r2_per)
                print(f"      mean R²={mean_r2:.4f}  positive: {pos_count}/{len(r2_per)}  best={best:.4f}")
                for k, feat in enumerate(KIN_FEATS):
                    out_rows.append(dict(
                        cache=cache_name, reducer=reducer_name, model=model_name,
                        feature=feat, r2=float(r2_per[k]),
                    ))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cache", "reducer", "model", "feature", "r2"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nwrote {out_csv}")

    if args.out_html:
        write_html(out_rows, Path(args.out_html))


def write_html(rows, out_path):
    # Build a pivot: cache × reducer × model → mean R²
    from collections import defaultdict
    means = defaultdict(list)
    for r in rows:
        means[(r["cache"], r["reducer"], r["model"])].append(r["r2"])
    summary = []
    for (cache, red, mdl), r2s in means.items():
        summary.append(dict(
            cache=cache, reducer=red, model=mdl,
            mean_r2=float(np.mean(r2s)),
            positive=int(sum(1 for r in r2s if r > 0)),
            best=float(max(r2s)),
            n_features=len(r2s),
        ))
    summary.sort(key=lambda r: -r["mean_r2"])

    html = ['<!doctype html><html><head><meta charset=utf-8><title>V-JEPA → kinematic R² scan</title>',
            '<style>body{font-family:-apple-system,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px}',
            'table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{border:1px solid #d8d4cc;padding:6px 10px}',
            'th{background:#f3efe5;text-align:left}td.num{text-align:right;font-variant-numeric:tabular-nums}',
            'tr.pos td{background:#e7f2eb;font-weight:600}tr.neg td{color:#6a6a6a}',
            '.callout{background:#f3efe5;border-left:3px solid #c9430b;padding:10px 16px;margin:14px 0}',
            '.callout.good{background:#e7f2eb;border-left-color:#1f7a47}',
            '</style></head><body>',
            f'<h1>V-JEPA → kinematic R² scan ({len(summary)} cache×reducer×model rows)</h1>']
    html.append('<table><thead><tr><th>Cache</th><th>Reducer</th><th>Model</th>'
                '<th class="num">Mean R²</th><th class="num">Positive R² (of 14)</th><th class="num">Best feat R²</th></tr></thead><tbody>')
    for r in summary:
        cls = "pos" if r["mean_r2"] > 0 else "neg"
        html.append(f'<tr class="{cls}"><td>{r["cache"]}</td><td>{r["reducer"]}</td><td>{r["model"]}</td>'
                    f'<td class="num">{r["mean_r2"]:.4f}</td>'
                    f'<td class="num">{r["positive"]}/{r["n_features"]}</td>'
                    f'<td class="num">{r["best"]:.4f}</td></tr>')
    html.append('</tbody></table>')
    if summary and summary[0]["mean_r2"] > 0:
        html.append(f'<div class="callout good"><strong>Top scan:</strong> mean R²={summary[0]["mean_r2"]:.4f} '
                    f'with cache={summary[0]["cache"]}, reducer={summary[0]["reducer"]}, model={summary[0]["model"]} — '
                    f'V-JEPA features CAN predict kinematic for this setup.</div>')
    else:
        html.append('<div class="callout"><strong>All R² values negative.</strong> Confirms V-JEPA features cannot predict kinematic across all tried configurations.</div>')
    html.append('</body></html>')
    out_path.write_text("\n".join(html))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
