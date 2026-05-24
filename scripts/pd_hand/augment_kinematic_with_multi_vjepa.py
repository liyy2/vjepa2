#!/usr/bin/env python
"""Augment kinematic CSVs with V-JEPA features from MULTIPLE caches concatenated."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
    if coverage is None:
        return list(range(len(fold_rows)))
    return [int(round(float(c.get("sample_index", -1)))) for c in coverage]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-train", action="append", required=True)
    p.add_argument("--cache-val", action="append", required=True)
    p.add_argument("--cache-prefix", action="append", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--kin-train-csv", required=True)
    p.add_argument("--kin-val-csv", required=True)
    p.add_argument("--out-train-csv", required=True)
    p.add_argument("--out-val-csv", required=True)
    p.add_argument("--pca-dim", type=int, default=16)
    p.add_argument("--n-fft-bands", type=int, default=3)
    args = p.parse_args()

    fold_tr_rows = list(csv.DictReader(open(args.fold_train_csv)))
    fold_va_rows = list(csv.DictReader(open(args.fold_val_csv)))

    # For each cache, compute summary features, build {clip_path: vec}.
    all_train_feats = {}  # clip_path → np.array
    all_val_feats = {}
    all_feat_names = []
    for cache_tr, cache_va, prefix in zip(args.cache_train, args.cache_val, args.cache_prefix):
        print(f"\n=== {prefix} ===")
        x_tr, m_tr, cov_tr = load_cache(cache_tr)
        x_va, m_va, cov_va = load_cache(cache_va)
        W, mu = fit_pca(x_tr, args.pca_dim)
        x_tr_p = apply_pca(x_tr, W, mu)
        x_va_p = apply_pca(x_va, W, mu)
        F_tr = summary_features(x_tr_p, m_tr, n_bands=args.n_fft_bands)
        F_va = summary_features(x_va_p, m_va, n_bands=args.n_fft_bands)
        print(f"  feat dim {F_tr.shape[1]}")
        feat_names = [f"{prefix}_{j}" for j in range(F_tr.shape[1])]
        all_feat_names.extend(feat_names)

        cache_to_fold_tr = map_cache_to_fold(cov_tr, fold_tr_rows)
        cache_to_fold_va = map_cache_to_fold(cov_va, fold_va_rows)
        for cache_idx, fold_idx in enumerate(cache_to_fold_tr):
            if 0 <= fold_idx < len(fold_tr_rows):
                clip = fold_tr_rows[fold_idx]["clip_path"]
                all_train_feats.setdefault(clip, {})
                for n, v in zip(feat_names, F_tr[cache_idx]):
                    all_train_feats[clip][n] = float(v)
        for cache_idx, fold_idx in enumerate(cache_to_fold_va):
            if 0 <= fold_idx < len(fold_va_rows):
                clip = fold_va_rows[fold_idx]["clip_path"]
                all_val_feats.setdefault(clip, {})
                for n, v in zip(feat_names, F_va[cache_idx]):
                    all_val_feats[clip][n] = float(v)

    print(f"\nTotal V-JEPA feature columns: {len(all_feat_names)}")

    def write_aug(in_path, out_path, by_clip):
        rows = list(csv.DictReader(open(in_path)))
        out_fields = list(rows[0].keys()) + all_feat_names
        kept = 0
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_fields)
            w.writeheader()
            for r in rows:
                clip = r["clip_path"]
                if clip in by_clip:
                    for n in all_feat_names:
                        r[n] = by_clip[clip].get(n, 0.0)
                    kept += 1
                else:
                    for n in all_feat_names:
                        r[n] = 0.0
                w.writerow(r)
        print(f"wrote {out_path}: {kept}/{len(rows)} with V-JEPA features")

    Path(args.out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val_csv).parent.mkdir(parents=True, exist_ok=True)
    write_aug(args.kin_train_csv, args.out_train_csv, all_train_feats)
    write_aug(args.kin_val_csv, args.out_val_csv, all_val_feats)


if __name__ == "__main__":
    main()
