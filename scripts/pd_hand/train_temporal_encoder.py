#!/usr/bin/env python
"""Cached-feature temporal-encoder trainer for PD hand-task severity probing.

Reads V-JEPA 2.1 spatially-pooled tubelet features from a .npz cache, builds a
stats-MLP head (mean/std/max/bin/velocity/per-segment-FFT stats over the
post-PCA per-tubelet sequence), and runs a grid of (seed, mixup, class-weight,
lr, head-shape) configurations. Saves per-config val probabilities so the
companion `cross_sweep_ensemble.py` can build greedy / exhaustive ensembles.

This is the final cleaned-up trainer — the v1/v2 trial code (TCN, GRU,
transformer model types, multiple FFT modes, many grid variants) has been
removed and only the winning configuration family is retained.

The best result on item 3.4 fold 0:
  - single config:   QWK 0.6519, acc 0.5606
  - greedy ensemble: QWK 0.7267, acc 0.5455 (K=7 from ~800 base models)
Kinematic baseline on same fold: QWK 0.7906, acc 0.6818.

Example:
  python train_temporal_encoder.py \\
      --train-npz <embeddings>/train_32f_step1_12seg_handcrop_nocrop_mean_concat.npz \\
      --val-npz   <embeddings>/val_32f_step1_12seg_handcrop_nocrop_mean_concat.npz \\
      --grid velocity --rank-by qwk --epochs 80 --patience 25 \\
      --out-dir <results>/sweep_velocity
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_EMB = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0"
)
DEFAULT_OUT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/fold_0_final"
)
NUM_CLASSES = 5


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    lr: float
    weight_decay: float
    dropout: float
    hidden_dim: int = 256
    d_model: int = 128
    temporal_bins: int = 24
    class_weight: str = "balanced"          # "balanced" or "none"
    loss: str = "ce"                        # "ce" or "corn"
    mixup_alpha: float = 0.2
    fft_bands: int = 4                      # 0 to disable
    fft_mode: str = "per_segment"           # "global" or "per_segment"
    segment_length: int = 16                # tubelets per V-JEPA segment (set per cache)
    use_velocity: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", default=f"{DEFAULT_EMB}/vjepa21_vitl384_train_32f_step1_12seg_handcrop_nocrop_mean_concat.npz")
    p.add_argument("--val-npz",   default=f"{DEFAULT_EMB}/vjepa21_vitl384_val_32f_step1_12seg_handcrop_nocrop_mean_concat.npz")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--segment-length", type=int, default=0,
                   help="If >0, override segment_length on all configs (e.g., 32 for 64-frame cache).")
    p.add_argument("--grid", default="velocity", choices=["velocity", "mega_seeds"])
    p.add_argument("--rank-by", default="qwk", choices=["acc", "qwk"])
    p.add_argument("--start-config", type=int, default=1, help="1-indexed inclusive")
    p.add_argument("--end-config", type=int, default=0, help="1-indexed inclusive; 0 means last")
    p.add_argument("--max-configs", type=int, default=0)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return p.parse_args()


_RANK_BY = "qwk"


def _rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    if _RANK_BY == "qwk":
        return (float(result["qwk"]), float(result["correct"]), -float(result["mae"]))
    return (float(result["correct"]), float(result["qwk"]), -float(result["mae"]))


# -----------------------------------------------------------------------------
# Grids (only the two that produced top results — velocity and mega_seeds)
# -----------------------------------------------------------------------------

def build_grid(name: str) -> list[TrainConfig]:
    if name == "velocity":
        return _velocity_grid()
    if name == "mega_seeds":
        return _mega_seeds_grid()
    raise ValueError(f"unknown grid {name}")


def _velocity_grid() -> list[TrainConfig]:
    """The grid that discovered the QWK 0.6519 single config.

    192 configs: 4 seeds × 2 lr × 2 head-shapes × 2 class-weight × 2 mixup × 2 loss
                 × {with FFT, no FFT}, all with velocity stats turned on.
    """
    configs: list[TrainConfig] = []
    seeds = [0, 17, 42, 98]
    for use_fft in [False, True]:
        for loss in ["ce", "corn"]:
            for mixup in [0.0, 0.2]:
                cw_iter = ["balanced", "none"] if loss == "ce" else ["none"]
                for class_weight in cw_iter:
                    for seed in seeds:
                        for lr in [1e-3, 3e-4]:
                            for hd_bins_drop in [(512, 12, 0.35), (256, 24, 0.5)]:
                                hd, bins, drop = hd_bins_drop
                                configs.append(TrainConfig(
                                    seed=seed, lr=lr, weight_decay=1e-2, dropout=drop,
                                    hidden_dim=hd, temporal_bins=bins,
                                    class_weight=class_weight,
                                    loss=loss, mixup_alpha=mixup,
                                    fft_bands=4 if use_fft else 0,
                                    fft_mode="per_segment",
                                    use_velocity=True,
                                ))
    return configs


def _mega_seeds_grid() -> list[TrainConfig]:
    """30 seeds × 6 top-known configurations — for ensemble seed diversity.

    The best cross-sweep ensembles drew their gains from seed diversity within
    a handful of good architectures, not from architectural variety. This grid
    runs the six best-known recipes against 30 seeds each (180 configs).
    """
    seeds = list(range(0, 30))
    configs: list[TrainConfig] = []
    base_specs = [
        # (loss, mixup, cw,         drop, hd,  bins, fft, fft_mode,      use_vel, lr)
        ("ce",   0.2, "balanced",   0.5,  256, 24,   0,   "per_segment", True,    3e-4),  # the QWK 0.6519 winner
        ("ce",   0.2, "balanced",   0.5,  256, 24,   4,   "per_segment", True,    3e-4),
        ("ce",   0.0, "balanced",   0.5,  256, 24,   0,   "per_segment", True,    3e-4),
        ("ce",   0.0, "balanced",   0.5,  256, 24,   0,   "global",      False,   1e-3),
        ("ce",   0.0, "none",       0.5,  256, 24,   0,   "global",      False,   1e-3),
        ("corn", 0.0, "none",       0.5,  256, 24,   0,   "global",      False,   1e-3),
    ]
    for seed in seeds:
        for (loss, mixup, cw, drop, hd, bins, fft_bands, fft_mode, use_vel, lr) in base_specs:
            configs.append(TrainConfig(
                seed=seed, lr=lr, weight_decay=1e-2, dropout=drop,
                hidden_dim=hd, temporal_bins=bins,
                class_weight=cw, loss=loss, mixup_alpha=mixup,
                fft_bands=fft_bands, fft_mode=fft_mode,
                use_velocity=use_vel,
            ))
    return configs


# -----------------------------------------------------------------------------
# Model — stats_mlp only (other model types were dominated in our sweeps)
# -----------------------------------------------------------------------------

class TemporalStatsMLP(nn.Module):
    """Per-tubelet projection → 30+ pooled stats over time → 2-layer MLP.

    Stats:
      - global: mean, std, max, first, last, (last-first)                  (6 vectors)
      - bin:    mean per bin, std per bin                                  (2*B vectors)
      - velocity (optional): mean|Δx|, std|Δx|, max|Δx|, peak-count proxy  (4 vectors)
      - FFT (optional): per-band log-magnitude (global or per-segment)     (K or 2K vectors)
    """

    def __init__(self, input_dim, num_out, hidden_dim, dropout, temporal_bins, d_model,
                 fft_bands=0, fft_mode="global", segment_length=16, use_velocity=False):
        super().__init__()
        self.temporal_bins = int(temporal_bins)
        self.fft_bands = int(fft_bands)
        self.fft_mode = fft_mode
        self.segment_length = int(segment_length)
        self.use_velocity = bool(use_velocity)

        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        fft_extra = 0
        if self.fft_bands > 0:
            fft_extra = self.fft_bands if self.fft_mode == "global" else 2 * self.fft_bands
        vel_extra = 4 if self.use_velocity else 0
        feature_dim = d_model * (6 + 2 * self.temporal_bins + fft_extra + vel_extra)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_out),
        )

    def forward(self, x):
        x = self.proj(x)
        chunks = torch.chunk(x, self.temporal_bins, dim=1)
        bin_mean = [c.mean(dim=1) for c in chunks]
        bin_std = [c.std(dim=1, unbiased=False) for c in chunks]
        features = [
            x.mean(dim=1), x.std(dim=1, unbiased=False), x.max(dim=1).values,
            x[:, 0], x[:, -1], x[:, -1] - x[:, 0],
            *bin_mean, *bin_std,
        ]
        if self.use_velocity:
            delta = x[:, 1:] - x[:, :-1]
            absd = delta.abs()
            features.append(absd.mean(dim=1))
            features.append(absd.std(dim=1, unbiased=False))
            features.append(absd.max(dim=1).values)
            thr = absd.mean(dim=1, keepdim=True) + absd.std(dim=1, keepdim=True, unbiased=False)
            features.append((absd > thr).float().mean(dim=1))
        if self.fft_bands > 0:
            if self.fft_mode == "global":
                x_c = x - x.mean(dim=1, keepdim=True)
                spec = torch.fft.rfft(x_c, dim=1, norm="ortho")
                mag = spec.abs()[:, 1:, :]
                n_freq = mag.shape[1]
                band_size = max(1, n_freq // self.fft_bands)
                for k in range(self.fft_bands):
                    lo, hi = k * band_size, ((k + 1) * band_size if k < self.fft_bands - 1 else n_freq)
                    features.append(torch.log1p(mag[:, lo:hi, :].mean(dim=1)))
            else:  # per_segment
                B, T_total, D = x.shape
                seg_len = self.segment_length
                n_seg = T_total // seg_len
                seg = x[:, : n_seg * seg_len].reshape(B, n_seg, seg_len, D)
                seg_c = seg - seg.mean(dim=2, keepdim=True)
                spec = torch.fft.rfft(seg_c, dim=2, norm="ortho")
                mag = spec.abs()[:, :, 1:, :]
                n_freq = mag.shape[2]
                band_size = max(1, n_freq // self.fft_bands)
                for k in range(self.fft_bands):
                    lo, hi = k * band_size, ((k + 1) * band_size if k < self.fft_bands - 1 else n_freq)
                    band = mag[:, :, lo:hi, :].mean(dim=2)
                    features.append(torch.log1p(band.mean(dim=1)))
                    features.append(torch.log1p(band.std(dim=1, unbiased=False)))
        return self.net(torch.cat(features, dim=1))


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def corn_loss(logits, labels, num_levels=NUM_CLASSES, pos_weight=None):
    """Cao et al. 2020 CORN — conditional BCE on K-1 cutpoint logits."""
    if logits.ndim != 2 or logits.shape[1] != num_levels - 1:
        raise ValueError(f"Expected logits [B, {num_levels - 1}], got {tuple(logits.shape)}")
    labels = labels.long()
    thresholds = torch.arange(num_levels - 1, device=labels.device)
    targets = (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()
    mask = (labels.unsqueeze(1) >= thresholds.unsqueeze(0)).float()
    if pos_weight is not None:
        pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)
        per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    else:
        per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)


def corn_probs_scores(logits):
    """K-1 cutpoint logits → (probs[B,K], expected_score[B])."""
    p_gt = torch.sigmoid(logits)
    p_gt_chain = torch.cumprod(p_gt, dim=1)
    B, Km1 = p_gt.shape
    K = Km1 + 1
    probs = logits.new_empty((B, K))
    probs[:, 0] = 1.0 - p_gt_chain[:, 0]
    for k in range(1, Km1):
        probs[:, k] = p_gt_chain[:, k - 1] - p_gt_chain[:, k]
    probs[:, -1] = p_gt_chain[:, -1]
    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs, p_gt_chain.sum(dim=1)


def corn_threshold_pos_weights(y, num_levels=NUM_CLASSES, device=None):
    arr = np.asarray(y, dtype=np.int64)
    weights = []
    for j in range(num_levels - 1):
        positives = int((arr > j).sum())
        negatives = int((arr <= j).sum())
        weights.append(float(negatives / positives) if positives > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def ce_class_weights(y, mode):
    if mode != "balanced":
        return None
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    nz = counts > 0
    weights = np.zeros_like(counts)
    weights[nz] = counts.sum() / (NUM_CLASSES * counts[nz])
    return torch.from_numpy(weights)


# -----------------------------------------------------------------------------
# Mixup
# -----------------------------------------------------------------------------

def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def single_loss(logits, y, cfg, ce_weights, corn_pos_weight):
    if cfg.loss == "ce":
        return F.cross_entropy(logits, y, weight=ce_weights)
    return corn_loss(logits, y, num_levels=NUM_CLASSES, pos_weight=corn_pos_weight)


def mix_loss(logits, y_a, y_b, lam, cfg, ce_weights, corn_pos_weight):
    if cfg.loss == "ce":
        loss_a = F.cross_entropy(logits, y_a, weight=ce_weights)
        loss_b = F.cross_entropy(logits, y_b, weight=ce_weights)
    else:
        loss_a = corn_loss(logits, y_a, num_levels=NUM_CLASSES, pos_weight=corn_pos_weight)
        loss_b = corn_loss(logits, y_b, num_levels=NUM_CLASSES, pos_weight=corn_pos_weight)
    return lam * loss_a + (1.0 - lam) * loss_b


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def load_npz(path):
    d = np.load(path)
    return d["x"].astype(np.float32), d["y"].astype(np.int64)


def normalize_from_train(x_train, x_val):
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = np.maximum(x_train.std(axis=(0, 1), keepdims=True), 1e-4)
    return (x_train - mean) / std, (x_val - mean) / std


def temporal_pca_from_train(x_train, x_val, pca_dim):
    if pca_dim >= x_train.shape[-1] or pca_dim <= 0:
        return x_train, x_val
    n_train, t, d = x_train.shape
    n_val = x_val.shape[0]
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0, whiten=False)
    train_red = pca.fit_transform(x_train.reshape(-1, d)).astype(np.float32)
    val_red = pca.transform(x_val.reshape(-1, d)).astype(np.float32)
    print(f"PCA {d}->{pca_dim}, explained_variance={pca.explained_variance_ratio_.sum():.5f}", flush=True)
    return train_red.reshape(n_train, t, pca_dim), val_red.reshape(n_val, t, pca_dim)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one_config(cfg, x_train, y_train, x_val, y_val, epochs, batch_size, device, patience):
    set_seed(cfg.seed)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator(); generator.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    num_out = NUM_CLASSES if cfg.loss == "ce" else (NUM_CLASSES - 1)
    model = TemporalStatsMLP(
        input_dim=x_train.shape[-1], num_out=num_out,
        hidden_dim=cfg.hidden_dim, dropout=cfg.dropout,
        temporal_bins=cfg.temporal_bins, d_model=cfg.d_model,
        fft_bands=cfg.fft_bands, fft_mode=cfg.fft_mode,
        segment_length=cfg.segment_length, use_velocity=cfg.use_velocity,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Linear warmup then constant. Cosine decay was tested but hurt — early-stop
    # was triggering before LR finished decaying, effectively training at a
    # shrinking LR most of the time. With this regime, warmup-then-hold gives
    # the model a stable late-epoch fine-tuning phase.
    steps_per_epoch = max(1, (len(train_ds) + batch_size - 1) // batch_size)
    warmup_steps = max(1, (epochs // 20)) * steps_per_epoch  # ~5% warmup

    def lr_at(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    global_step = 0

    weights = ce_class_weights(y_train, mode=cfg.class_weight) if cfg.loss == "ce" else None
    if weights is not None:
        weights = weights.to(device)
    corn_pos_weight = (
        corn_threshold_pos_weights(y_train, num_levels=NUM_CLASSES, device=device)
        if cfg.loss == "corn" else None
    )

    x_val_t = torch.from_numpy(x_val).to(device)
    best: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = cfg.lr * lr_at(global_step)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            if cfg.mixup_alpha > 0:
                xb, ya, yb_p, lam = mixup_batch(xb, yb, alpha=cfg.mixup_alpha)
                logits = model(xb)
                loss = mix_loss(logits, ya, yb_p, lam, cfg, weights, corn_pos_weight)
            else:
                logits = model(xb)
                loss = single_loss(logits, yb, cfg, weights, corn_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1

        model.eval()
        with torch.inference_mode():
            logits = model(x_val_t)
            if cfg.loss == "corn":
                probs, _ = corn_probs_scores(logits)
                probs_np = probs.cpu().numpy().astype(np.float32)
            else:
                probs_np = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            preds = probs_np.argmax(axis=1).astype(np.int64)
        result = _metrics(y_val, preds)
        result["epoch"] = epoch
        if best is None or _rank_key(result) > _rank_key(best):
            best = result
            best["val_probs"] = probs_np.tolist()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    assert best is not None
    best["config"] = asdict(cfg)
    best["epochs_ran"] = epoch
    return best


def _metrics(labels, pred):
    return {
        "correct": int((pred == labels).sum()),
        "val_acc": float(accuracy_score(labels, pred)),
        "qwk": float(cohen_kappa_score(labels, pred, weights="quadratic")),
        "mae": float(mean_absolute_error(labels, pred)),
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(NUM_CLASSES))).tolist(),
        "predictions": pred.tolist(),
        "labels": labels.tolist(),
    }


def build_topk_ensembles(results, labels):
    ensembles = []
    probs = [np.asarray(r["val_probs"], dtype=np.float32) for r in results if "val_probs" in r]
    if len(probs) < 2:
        return ensembles
    running = np.zeros_like(probs[0])
    for idx, p in enumerate(probs[: min(30, len(probs))], start=1):
        running += p
        if idx < 2:
            continue
        avg = running / idx
        pred = avg.argmax(axis=1).astype(np.int64)
        m = _metrics(labels, pred)
        m["ensemble_top_k"] = idx
        m["config"] = {"top_k": idx, "kind": "topk_prob_ensemble"}
        ensembles.append(m)
    return sorted(ensembles, key=_rank_key, reverse=True)


def maybe_init_wandb(args, train_shape, val_shape, num_configs):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            config={
                "train_npz": args.train_npz, "val_npz": args.val_npz,
                "train_shape": list(train_shape), "val_shape": list(val_shape),
                "epochs": args.epochs, "batch_size": args.batch_size,
                "num_configs": num_configs, "grid": args.grid,
                "rank_by": args.rank_by, "pca_dim": args.pca_dim,
            },
        )
    except Exception as exc:
        print(f"wandb init failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def main():
    args = parse_args()
    global _RANK_BY
    _RANK_BY = args.rank_by
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_npz(args.train_npz)
    x_val, y_val = load_npz(args.val_npz)
    raw_train_shape, raw_val_shape = x_train.shape, x_val.shape
    x_train, x_val = normalize_from_train(x_train, x_val)
    if args.pca_dim > 0:
        x_train, x_val = temporal_pca_from_train(x_train, x_val, args.pca_dim)

    all_configs = build_grid(args.grid)
    if args.segment_length > 0:
        all_configs = [replace(cfg, segment_length=args.segment_length) for cfg in all_configs]
    start = max(1, args.start_config)
    end = args.end_config if args.end_config > 0 else len(all_configs)
    configs = all_configs[start - 1 : end]
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    wandb_run = maybe_init_wandb(args, x_train.shape, x_val.shape, len(configs))
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    partial_path = out_dir / "temporal_encoder_partial_results.json"
    for local_idx, cfg in enumerate(configs, start=1):
        idx = start + local_idx - 1
        print(f"config {idx}/{len(all_configs)} {cfg}", flush=True)
        result = train_one_config(
            cfg=cfg,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            epochs=args.epochs, batch_size=args.batch_size,
            device=device, patience=args.patience,
        )
        result["config_index"] = idx
        results.append(result)
        if best is None or _rank_key(result) > _rank_key(best):
            best = result
            print(
                f"new_best {best['correct']}/{len(y_val)} "
                f"acc={best['val_acc']:.4f} qwk={best['qwk']:.4f} mae={best['mae']:.4f} "
                f"cfg={best['config']}",
                flush=True,
            )
        partial = sorted(results, key=_rank_key, reverse=True)
        partial_path.write_text(json.dumps({"best": partial[0], "top_results": partial[:50]}, indent=2))
        if wandb_run is not None:
            wandb_run.log({
                "config_index": idx,
                "config_val_acc": result["val_acc"],
                "config_correct": result["correct"],
                "config_qwk": result["qwk"],
                "config_mae": result["mae"],
                "best_val_acc": partial[0]["val_acc"],
                "best_qwk": partial[0]["qwk"],
                "best_mae": partial[0]["mae"],
            })

    results = sorted(results, key=_rank_key, reverse=True)
    ensembles = build_topk_ensembles(results, y_val)
    overall = sorted(results + ensembles, key=_rank_key, reverse=True)
    output = {
        "note": "Pure-vision temporal encoder on cached V-JEPA 2.1 tubelet features.",
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "raw_train_shape": list(raw_train_shape),
        "raw_val_shape": list(raw_val_shape),
        "pca_dim": args.pca_dim,
        "grid": args.grid,
        "rank_by": args.rank_by,
        "best": overall[0],
        "best_single_model": results[0],
        "best_ensemble": ensembles[0] if ensembles else None,
        "top_results": results[:50],
        "top_ensembles": ensembles[:20],
    }
    out_path = out_dir / "temporal_encoder_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}")
    print(
        f"BEST {overall[0]['correct']}/{len(y_val)} "
        f"acc={overall[0]['val_acc']:.4f} qwk={overall[0]['qwk']:.4f} mae={overall[0]['mae']:.4f}"
    )
    if wandb_run is not None:
        wandb_run.log({
            "final_best_val_acc": overall[0]["val_acc"],
            "final_best_qwk": overall[0]["qwk"],
            "final_best_mae": overall[0]["mae"],
        })
        wandb_run.finish()


if __name__ == "__main__":
    main()
