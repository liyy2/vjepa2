#!/usr/bin/env python
"""Augment kinematic feature CSVs with V-JEPA cache-derived summary features.

Takes a V-JEPA cache (.npz with x, y, mask, coverage) plus the kinematic baseline's
train/val feature CSVs, computes per-clip V-JEPA summary statistics + PCA components +
FFT band powers (those that had positive R²), aligns by sample_index → fold-CSV row
→ clip_path, and writes augmented CSVs with new <prefix>_<j> columns appended.

The augmented CSVs can then be passed to the existing kinematic baseline runner
(train_finger_tapping_kinematic_baseline.py) which fits ExtraTrees on all numeric
columns. This yields a clean hybrid model comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-train", required=True)
    p.add_argument("--cache-val", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--out-train-csv", required=True)
    p.add_argument("--out-val-csv", required=True)
    p.add_argument("--prefix", default="vjepa")
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--n-fft-bands", type=int, default=3)
    return p.parse_args()


def load_cache(p):
    d = np.load(p, allow_pickle=False)
    x = d["x"].astype(np.float32)
    mask = d["mask"].astype(bool) if "mask" in d.files else np.ones(x.shape[:2], dtype=bool)
    coverage = [json.loads(str(s)) for s in d["coverage"]] if "coverage" in d.files else None
    return x, mask, coverage


def fit_pca(x_train, dim):
    from sklearn.decomposition import PCA
    flat = x_train.reshape(-1, x_train.shape[-1])
    mu = flat.mean(axis=0)
    pca = PCA(n_components=min(dim, flat.shape[1]), random_state=0)
    pca.fit(flat - mu)
    return pca.components_.astype(np.float32), mu.astype(np.float32)


def apply_pca(x, W, mu):
    flat = (x.reshape(-1, x.shape[-1]) - mu) @ W.T
    return flat.reshape(x.shape[0], x.shape[1], -1)


def summary_features(x, mask, fps_per_tubelet=15.0, n_bands=3):
    """[N, T, D] → [N, F]: mean/std/|Δ|/first/last/Δ + FFT band powers."""
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
    # FFT bands (we know 0.5-2, 2-5, 5-8 are partially predictable)
    centered = (x - mean[:, None]) * valid
    spec = np.fft.rfft(centered, axis=1, norm="ortho")
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fps_per_tubelet)
    bands_hz = [(0.5, 2.0), (2.0, 5.0), (5.0, 8.0)][:n_bands]
    band_outs = []
    for lo, hi in bands_hz:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            band_outs.append(np.zeros((x.shape[0], x.shape[2]), dtype=np.float32))
        else:
            band_outs.append(np.log1p(mag[:, idx].mean(axis=1)).astype(np.float32))
    band = np.concatenate(band_outs, axis=1)
    return np.concatenate([mean, std, vmean, band], axis=1).astype(np.float32)


def map_cache_to_fold(coverage, fold_rows):
    """Return list of length len(coverage), with the fold CSV row index for each cache row."""
    if coverage is None:
        return list(range(len(fold_rows)))
    return [int(round(float(c.get("sample_index", -1)))) for c in coverage]


def main():
    args = parse_args()

    x_tr, m_tr, cov_tr = load_cache(args.cache_train)
    x_va, m_va, cov_va = load_cache(args.cache_val)

    # PCA fit on train cache
    W, mu = fit_pca(x_tr, args.pca_dim)
    print(f"PCA fit: D={x_tr.shape[-1]} → {W.shape[0]}")
    x_tr_p = apply_pca(x_tr, W, mu)
    x_va_p = apply_pca(x_va, W, mu)

    # Per-clip summary features
    F_tr = summary_features(x_tr_p, m_tr, n_bands=args.n_fft_bands)
    F_va = summary_features(x_va_p, m_va, n_bands=args.n_fft_bands)
    print(f"V-JEPA augment feature dim: {F_tr.shape[1]} (train) / {F_va.shape[1]} (val)")

    # Map cache rows → fold CSV row index → clip_path
    fold_tr_rows = list(csv.DictReader(open(args.fold_train_csv)))
    fold_va_rows = list(csv.DictReader(open(args.fold_val_csv)))
    cache_to_fold_tr = map_cache_to_fold(cov_tr, fold_tr_rows)
    cache_to_fold_va = map_cache_to_fold(cov_va, fold_va_rows)

    feat_names = [f"{args.prefix}_{j}" for j in range(F_tr.shape[1])]

    # Build dict clip_path → feature vector
    def vjepa_by_clip(F, cache_to_fold, fold_rows):
        out = {}
        for cache_idx, fold_idx in enumerate(cache_to_fold):
            if 0 <= fold_idx < len(fold_rows):
                clip = fold_rows[fold_idx].get("clip_path")
                out[clip] = F[cache_idx]
        return out

    vj_by_clip_tr = vjepa_by_clip(F_tr, cache_to_fold_tr, fold_tr_rows)
    vj_by_clip_va = vjepa_by_clip(F_va, cache_to_fold_va, fold_va_rows)
    print(f"V-JEPA train coverage: {len(vj_by_clip_tr)} / {len(fold_tr_rows)}")
    print(f"V-JEPA val   coverage: {len(vj_by_clip_va)} / {len(fold_va_rows)}")

    # Augment kinematic CSV
    def augment_csv(in_path, out_path, vj_by_clip):
        rows = list(csv.DictReader(open(in_path)))
        if not rows:
            return
        out_fields = list(rows[0].keys()) + feat_names
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            kept = 0
            for r in rows:
                clip = r.get("clip_path")
                if clip in vj_by_clip:
                    vec = vj_by_clip[clip]
                    for j, name in enumerate(feat_names):
                        r[name] = float(vec[j])
                    kept += 1
                else:
                    for name in feat_names:
                        r[name] = 0.0
                w.writerow(r)
        print(f"wrote {out_path}: {kept}/{len(rows)} with V-JEPA features")

    Path(args.out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val_csv).parent.mkdir(parents=True, exist_ok=True)
    augment_csv(args.kin_train_csv, args.out_train_csv, vj_by_clip_tr)
    augment_csv(args.kin_val_csv, args.out_val_csv, vj_by_clip_va)


if __name__ == "__main__":
    main()
