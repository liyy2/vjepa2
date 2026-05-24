#!/usr/bin/env python
"""LightGBM on hand-crafted summary statistics of V-JEPA temporal token sequences.

Same recipe as the kinematic baseline (MediaPipe → handcrafted features → LightGBM)
but applied to the cached V-JEPA temporal embedding sequence instead. With only 404
train clips and high-dimensional features, gradient boosting tends to be more
sample-efficient than transformers.

Features per cached [T, D] clip, per dim d:
  - mean, std, min, max, p10, p50, p90
  - velocity: mean|Δ|, std|Δ|, max|Δ|
  - acceleration: mean|Δ²|, std|Δ²|
  - FFT band power (log) in 2 bands: 0.5–2 Hz, 2–6 Hz (covers PD finger-tap range)
  - first half mean, last half mean, decrement (last-first)
  - peak count proxy (count |Δ| > mean+std)

Plus 2-stream version: same features on fine and coarse caches, concatenated.

D=1024 × 16 features = 16384-dim. Top-K PCA before LightGBM to keep variance low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", action="append", required=True)
    p.add_argument("--val-npz", action="append", required=True)
    p.add_argument("--scale-names", action="append", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="Per-stream PCA dimensionality on raw embeddings before computing summary stats.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Inferred per-tubelet sample rate is fps / tubelet_size; default 30/2=15 Hz.")
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--n-bands", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lgbm-objective", default="multiclass",
                   choices=["multiclass", "regression"])
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--feature-fraction", type=float, default=0.6)
    p.add_argument("--bagging-fraction", type=float, default=0.85)
    return p.parse_args()


def load(path: str):
    d = np.load(path)
    return (
        d["x"].astype(np.float32),
        d["y"].astype(np.int64),
        d["mask"].astype(bool) if "mask" in d.files else np.ones(d["x"].shape[:2], dtype=bool),
    )


def fit_pca(x_train: np.ndarray, dim: int):
    """Fit PCA on flattened [N*T, D] features.
    Returns (W [dim,D], mu [D]).
    """
    from sklearn.decomposition import PCA
    N, T, D = x_train.shape
    flat = x_train.reshape(-1, D)
    mu = flat.mean(axis=0)
    pca = PCA(n_components=dim, random_state=0)
    pca.fit(flat - mu)
    return pca.components_.astype(np.float32), mu.astype(np.float32)


def apply_pca(x: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
    N, T, D = x.shape
    flat = (x.reshape(-1, D) - mu) @ W.T
    return flat.reshape(N, T, -1)


def fft_band_power(seq: np.ndarray, mask: np.ndarray, fps: float, bands_hz: list[tuple[float, float]]):
    """seq [B, T, D], mask [B, T]; returns log band-power [B, len(bands_hz), D]."""
    B, T, D = seq.shape
    # zero out masked tokens
    valid = mask.astype(np.float32)[..., None]
    valid_count = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    means = (seq * valid).sum(axis=1, keepdims=True) / valid_count[..., None]
    centered = (seq - means) * valid
    spec = np.fft.rfft(centered, axis=1, norm="ortho")
    mag = np.abs(spec)  # [B, T/2+1, D]
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    out = np.zeros((B, len(bands_hz), D), dtype=np.float32)
    for k, (lo, hi) in enumerate(bands_hz):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            continue
        out[:, k] = mag[:, idx].mean(axis=1)
    return np.log1p(out)


def summary_features(seq: np.ndarray, mask: np.ndarray, fps: float, n_bands: int) -> np.ndarray:
    """Compute per-clip summary statistics. seq [N, T, D], mask [N, T] → [N, D*F]."""
    N, T, D = seq.shape
    valid = mask.astype(np.float32)[..., None]
    counts = mask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)

    # Global stats per dim
    mean = (seq * valid).sum(axis=1) / counts
    var = (((seq - mean[:, None]) ** 2) * valid).sum(axis=1) / counts
    std = np.sqrt(np.clip(var, 0, None))
    # Percentiles with NaN-fill of invalid tokens
    seq_for_pctl = np.where(mask[..., None], seq, np.nan)
    p10 = np.nanpercentile(seq_for_pctl, 10, axis=1)
    p50 = np.nanpercentile(seq_for_pctl, 50, axis=1)
    p90 = np.nanpercentile(seq_for_pctl, 90, axis=1)
    # max/min over valid
    seq_big = np.where(mask[..., None], seq, -np.inf)
    mx = seq_big.max(axis=1)
    seq_small = np.where(mask[..., None], seq, np.inf)
    mn = seq_small.min(axis=1)

    # Velocity
    delta = seq[:, 1:] - seq[:, :-1]
    dmask = (mask[:, 1:] & mask[:, :-1])
    dvalid = dmask.astype(np.float32)[..., None]
    dcounts = dmask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    absd = np.abs(delta)
    vmean = (absd * dvalid).sum(axis=1) / dcounts
    vstd = np.sqrt((((absd - vmean[:, None]) ** 2) * dvalid).sum(axis=1) / dcounts).clip(min=0)
    absd_big = np.where(dmask[..., None], absd, -np.inf)
    vmax = absd_big.max(axis=1)

    # Acceleration
    accel = delta[:, 1:] - delta[:, :-1]
    amask = (dmask[:, 1:] & dmask[:, :-1])
    aabs = np.abs(accel)
    avalid = amask.astype(np.float32)[..., None]
    acounts = amask.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    amean = (aabs * avalid).sum(axis=1) / acounts
    astd = np.sqrt((((aabs - amean[:, None]) ** 2) * avalid).sum(axis=1) / acounts).clip(min=0)

    # Decrement: difference of last quarter mean vs first quarter mean
    Q = max(1, T // 4)
    first_q = (seq[:, :Q] * valid[:, :Q]).sum(axis=1) / mask[:, :Q].sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    last_q = (seq[:, -Q:] * valid[:, -Q:]).sum(axis=1) / mask[:, -Q:].sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)
    decrement = last_q - first_q

    # FFT bands
    bands_hz = [(0.5, 2.0), (2.0, 6.0)] if n_bands == 2 else [(0.5, 2.0), (2.0, 4.0), (4.0, 6.0)]
    fft = fft_band_power(seq, mask, fps, bands_hz)  # [N, n_bands, D]
    fft_flat = fft.reshape(N, -1)

    # Peak rate proxy: fraction of velocity points where |Δ| exceeds mean+std
    thr = (vmean + vstd)[:, None]
    peak_rate = ((absd > thr) * dvalid).sum(axis=1) / dcounts

    feats = np.concatenate(
        [mean, std, p10, p50, p90, mx, mn, vmean, vstd, vmax, amean, astd, decrement, first_q, last_q, peak_rate],
        axis=1,
    )
    feats = np.concatenate([feats, fft_flat], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def qwk(y, p, n_cls=NUM_CLASSES) -> float:
    cm = np.zeros((n_cls, n_cls), dtype=np.float64)
    for yi, pi in zip(y, p):
        cm[int(yi), int(pi)] += 1
    w = np.zeros_like(cm)
    for i in range(n_cls):
        for j in range(n_cls):
            w[i, j] = (i - j) ** 2 / (n_cls - 1) ** 2
    h_y = cm.sum(axis=1, keepdims=True)
    h_p = cm.sum(axis=0, keepdims=True)
    n = cm.sum()
    if n == 0:
        return 0.0
    exp = (h_y @ h_p) / n
    num = (w * cm).sum()
    den = (w * exp).sum()
    return 1 - num / (den + 1e-9)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.scale_names:
        args.scale_names = [f"scale{i}" for i in range(len(args.train_npz))]

    sample_rate = args.fps / float(args.tubelet_size)
    print(f"sample rate per tubelet token: {sample_rate:.2f} Hz")

    # Load each scale, fit PCA on train, apply to both
    train_feats = []
    val_feats = []
    y_train_ref = None
    y_val_ref = None
    for tp, vp, nm in zip(args.train_npz, args.val_npz, args.scale_names):
        print(f"loading scale {nm} from {tp}")
        x_tr, y_tr, m_tr = load(tp)
        x_va, y_va, m_va = load(vp)
        if y_train_ref is None:
            y_train_ref, y_val_ref = y_tr, y_va
        else:
            assert np.array_equal(y_train_ref, y_tr), f"train label mismatch for {tp}"
            assert np.array_equal(y_val_ref, y_va), f"val label mismatch for {vp}"

        W, mu = fit_pca(x_tr, args.pca_dim)
        x_tr_p = apply_pca(x_tr, W, mu)
        x_va_p = apply_pca(x_va, W, mu)
        ft = summary_features(x_tr_p, m_tr, sample_rate, args.n_bands)
        fv = summary_features(x_va_p, m_va, sample_rate, args.n_bands)
        print(f"  {nm}: train feat shape {ft.shape}, val feat shape {fv.shape}")
        train_feats.append(ft)
        val_feats.append(fv)

    X_tr = np.concatenate(train_feats, axis=1)
    X_va = np.concatenate(val_feats, axis=1)
    print(f"final X_train shape: {X_tr.shape}; X_val shape: {X_va.shape}")

    import lightgbm as lgb

    if args.lgbm_objective == "multiclass":
        params = dict(
            objective="multiclass",
            num_class=NUM_CLASSES,
            metric="multi_logloss",
            num_leaves=args.num_leaves,
            learning_rate=args.learning_rate,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=3,
            seed=args.seed,
            verbose=-1,
        )
    else:
        params = dict(
            objective="regression",
            metric="rmse",
            num_leaves=args.num_leaves,
            learning_rate=args.learning_rate,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=3,
            seed=args.seed,
            verbose=-1,
        )

    tr_set = lgb.Dataset(X_tr, label=y_train_ref.astype(np.float64))
    va_set = lgb.Dataset(X_va, label=y_val_ref.astype(np.float64), reference=tr_set)

    model = lgb.train(
        params=params,
        train_set=tr_set,
        valid_sets=[va_set],
        num_boost_round=args.n_estimators,
        callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=50)],
    )

    if args.lgbm_objective == "multiclass":
        prob = model.predict(X_va, num_iteration=model.best_iteration)
        yhat = prob.argmax(axis=1)
    else:
        cont = model.predict(X_va, num_iteration=model.best_iteration)
        yhat = np.clip(np.round(cont), 0, NUM_CLASSES - 1).astype(np.int64)

    acc = float((yhat == y_val_ref).mean())
    qwk_score = float(qwk(y_val_ref, yhat))
    mae = float(np.abs(yhat - y_val_ref).mean())
    print(f"VAL  acc={acc:.4f}  qwk={qwk_score:.4f}  mae={mae:.4f}")

    result = {
        "acc": acc,
        "qwk": qwk_score,
        "mae": mae,
        "best_iteration": int(model.best_iteration or args.n_estimators),
        "n_train": int(len(y_train_ref)),
        "n_val": int(len(y_val_ref)),
        "args": vars(args),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
