#!/usr/bin/env python
"""Train an order-aware cross-window CORN combiner on cached V-JEPA features.

Goal: replace the two aggressive defaults in the current eval pipeline
  - mean of segment logits across the multi-clip grid (commutative, throws ordering)
  - bag-of-tokens stats MLP (no positional information)
with a transformer that operates over the full spatially-pooled temporal
token sequence (e.g. 12 segments × 16 tubelets = 192 tokens for the dense cache),
optionally fusing multiple temporal resolutions ("fine" + "coarse" tubelet caches),
and emits CORN logits.

Cached input shape per clip is [T, D] from
  scripts/pd_hand/cache_vjepa21_temporal_embeddings.py
where the cache already collapses the 576 spatial patches to a single D-dim
vector per tubelet.  This script does NOT discard temporal order: it uses 2D
positional embeddings ((segment_idx, within_segment_idx)) and learns the
cross-window combiner directly from the labeled fold-0 train set.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

NUM_CLASSES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-npz",
        action="append",
        required=True,
        help="One or more paths; pass multiple times for multi-scale fusion (e.g. fine + coarse).",
    )
    p.add_argument("--val-npz", action="append", required=True)
    p.add_argument("--scale-names", action="append", default=None,
                   help="Optional label per scale, used only for logs. Must match --train-npz/--val-npz count.")
    p.add_argument("--tubelets-per-segment", action="append", type=int, default=None,
                   help="One per scale; e.g. --tubelets-per-segment 16 16 for two scales. "
                        "Defaults inferred from cache config when possible.")
    p.add_argument("--num-segments-for-pos", action="append", type=int, default=None,
                   help="One per scale; e.g. --num-segments-for-pos 12 8.")
    p.add_argument("--out-dir", required=True)

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="Upper bound on total token positions across all scales.")
    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--class-weight", choices=["none", "balanced", "sqrt", "manual"], default="balanced")
    p.add_argument("--manual-class-weights", default="",
                   help='Comma-separated list of 5 weights, used when --class-weight manual.')
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Beta(alpha, alpha) mixup on the token features when > 0. Mixes both x and y in CORN binary level space.")
    p.add_argument("--token-dropout", type=float, default=0.0,
                   help="Random per-token dropout during training (drops entire (B, t) tokens by zeroing mask).")
    p.add_argument("--segment-dropout", type=float, default=0.0,
                   help="Random per-segment dropout: drop a whole segment of tubelets uniformly. Acts at the level of the multi-clip eval grid.")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--select-by", choices=["qwk", "acc", "qwk_acc"], default="qwk")
    p.add_argument("--rank-by", default="qwk")
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--cosine-epochs", type=int, default=0,
                   help="Number of post-warmup epochs over which to cosine-decay LR to final. "
                        "After this, LR stays at final (= base_lr * final_lr_ratio). 0 = use total epochs.")
    p.add_argument("--final-lr-ratio", type=float, default=0.05,
                   help="final_lr / base_lr ratio at end of cosine schedule.")
    p.add_argument("--distill-train-csv", default="",
                   help="Optional kinematic feature CSV for auxiliary regression distillation (train split). "
                        "Joined to the cache via clip_path → fold-split CSV row → cache sample_index.")
    p.add_argument("--distill-val-csv", default="")
    p.add_argument("--distill-fold-train-csv", default="",
                   help="The fold_<i>_train.csv that the V-JEPA cache was built from (needed to map sample_index to clip_path).")
    p.add_argument("--distill-fold-val-csv", default="")
    p.add_argument("--distill-features", default=
                   "bbox_area_peak_rate_hz,"
                   "bbox_area_peak_interval_cv,"
                   "bbox_area_decrement,"
                   "bbox_area_cycle_amplitude_mean,"
                   "bbox_area_cycle_amplitude_std,"
                   "bbox_area_dominant_freq_hz,"
                   "bbox_area_bandpower_2_5_hz,"
                   "bbox_area_velocity_mean_abs,"
                   "bbox_area_peak_count",
                   help="Comma-separated list of kinematic columns to predict.")
    p.add_argument("--distill-weight", type=float, default=0.3,
                   help="Auxiliary distillation loss weight.")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-mode", default="online")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ScaleCache:
    x_train: np.ndarray  # [N, T_i, D_i]
    mask_train: np.ndarray  # [N, T_i]
    x_val: np.ndarray
    mask_val: np.ndarray
    name: str
    num_segments_for_pos: int
    tubelets_per_segment: int


def load_distillation_targets(
    fold_csv: str,
    distill_csv: str,
    feature_cols: list[str],
    cache_coverage: list[dict] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_aux [N, F], aux_mask [N]) in the order matching the V-JEPA cache.

    fold_csv is the splits/fold_<i>_{train,val}.csv used to build the cache.
    distill_csv is the kinematic feature CSV (has clip_path + the feature cols).
    cache_coverage is the parsed coverage metadata from the V-JEPA cache
    (each row has 'sample_index' = CSV row index for the cached order).
    """
    import csv
    fold_rows = list(csv.DictReader(open(fold_csv)))
    # Map clip_path → kinematic features
    kin_rows = list(csv.DictReader(open(distill_csv)))
    by_path: dict[str, dict] = {r.get("clip_path"): r for r in kin_rows if r.get("clip_path")}
    if cache_coverage is None:
        # Cache is assumed to be in CSV natural order
        sample_indices = list(range(len(fold_rows)))
    else:
        sample_indices = [int(round(float(c.get("sample_index", -1)))) for c in cache_coverage]
    N = len(sample_indices)
    F = len(feature_cols)
    y_aux = np.full((N, F), np.nan, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)
    for i, si in enumerate(sample_indices):
        if si < 0 or si >= len(fold_rows):
            continue
        clip = fold_rows[si].get("clip_path")
        kin = by_path.get(clip)
        if kin is None:
            continue
        vals = []
        ok = True
        for col in feature_cols:
            v = kin.get(col, "")
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    ok = False
                    break
                vals.append(fv)
            except Exception:
                ok = False
                break
        if not ok:
            continue
        y_aux[i] = np.asarray(vals, dtype=np.float32)
        mask[i] = True
    return y_aux, mask


def standardize_aux(y_train: np.ndarray, m_train: np.ndarray,
                    y_val: np.ndarray, m_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center+scale each aux column using train statistics; leave NaN rows as 0 (masked)."""
    valid = m_train
    if not valid.any():
        return y_train, y_val, np.zeros(y_train.shape[-1]), np.ones(y_train.shape[-1])
    mu = np.nanmean(y_train[valid], axis=0)
    sd = np.nanstd(y_train[valid], axis=0) + 1e-6
    yt = np.where(np.isnan(y_train), 0.0, (y_train - mu) / sd)
    yv = np.where(np.isnan(y_val), 0.0, (y_val - mu) / sd)
    return yt.astype(np.float32), yv.astype(np.float32), mu, sd


def load_scale(train_path: str, val_path: str, name: str | None,
               num_segments_for_pos: int | None,
               tubelets_per_segment: int | None) -> ScaleCache:
    train = np.load(train_path)
    val = np.load(val_path)
    x_tr = train["x"].astype(np.float32)
    x_va = val["x"].astype(np.float32)
    y_tr = train["y"].astype(np.int64)
    y_va = val["y"].astype(np.int64)
    mask_tr = train["mask"].astype(bool) if "mask" in train.files else np.ones(x_tr.shape[:2], dtype=bool)
    mask_va = val["mask"].astype(bool) if "mask" in val.files else np.ones(x_va.shape[:2], dtype=bool)

    cfg_tr = json.loads(str(train["config"])) if "config" in train.files else {}
    cfg_va = json.loads(str(val["config"])) if "config" in val.files else {}
    data_cfg = cfg_tr.get("data", {}) or cfg_va.get("data", {})
    inferred_num_seg = int(data_cfg.get("num_segments", 12))
    fpc = int(data_cfg.get("frames_per_clip", 32))
    tubelet_size = 2  # V-JEPA 2.1 default
    inferred_tubelets = max(1, fpc // tubelet_size)

    return ScaleCache(
        x_train=x_tr,
        mask_train=mask_tr,
        x_val=x_va,
        mask_val=mask_va,
        name=name or Path(train_path).stem,
        num_segments_for_pos=int(num_segments_for_pos) if num_segments_for_pos else inferred_num_seg,
        tubelets_per_segment=int(tubelets_per_segment) if tubelets_per_segment else inferred_tubelets,
    ), y_tr, y_va


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TwoDPos(nn.Module):
    """Learned 2D positional encoding for (segment, within-segment) indices.

    A 1D embedding for segment index and another for within-segment tubelet
    index, summed and added to the projected features.
    """

    def __init__(self, d_model: int, max_segments: int, max_within: int):
        super().__init__()
        self.seg = nn.Embedding(max_segments, d_model)
        self.within = nn.Embedding(max_within, d_model)
        nn.init.trunc_normal_(self.seg.weight, std=0.02)
        nn.init.trunc_normal_(self.within.weight, std=0.02)
        self.max_segments = max_segments
        self.max_within = max_within

    def forward(self, T: int, tubelets_per_segment: int, device) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        seg_idx = (idx // tubelets_per_segment).clamp_max(self.max_segments - 1)
        within_idx = (idx % tubelets_per_segment).clamp_max(self.max_within - 1)
        return self.seg(seg_idx) + self.within(within_idx)


class TemporalCornCombiner(nn.Module):
    """Cross-window CORN combiner over the full temporal token sequence.

    Forward expects a list (one entry per scale) of (x: [B, T_i, D_i], mask: [B, T_i]).
    Each scale is projected to d_model, gets a stream embedding plus a 2D positional
    encoding ((segment, within-segment) for that scale), and all scales are
    concatenated along the time axis before the transformer encoder.  A learnable
    CLS token reads out the clip representation.  Output: CORN binary logits
    over (num_classes - 1) levels.

    Optionally has an auxiliary regression head off the CLS token, used for
    knowledge distillation against scalar kinematic features.
    """

    def __init__(
        self,
        scales: Sequence[ScaleCache],
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_mult: int,
        dropout: float,
        num_classes: int = NUM_CLASSES,
        aux_dim: int = 0,
    ):
        super().__init__()
        self.n_scales = len(scales)
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(int(s.x_train.shape[-1])),
                nn.Linear(int(s.x_train.shape[-1]), d_model),
                nn.GELU(),
            )
            for s in scales
        ])
        # Per-scale 2D positional encodings, sized for the worst case
        self.pos = nn.ModuleList([
            TwoDPos(d_model,
                    max_segments=max(64, s.num_segments_for_pos),
                    max_within=max(32, s.tubelets_per_segment))
            for s in scales
        ])
        # Stream embeddings let the transformer distinguish fine vs coarse tokens
        self.stream_emb = nn.Parameter(torch.zeros(self.n_scales, d_model))
        nn.init.trunc_normal_(self.stream_emb, std=0.02)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes - 1)
        self.aux_head = nn.Linear(d_model, aux_dim) if aux_dim > 0 else None
        self._scale_tubelets = [int(s.tubelets_per_segment) for s in scales]

    def forward(
        self,
        scale_inputs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = scale_inputs[0][0].shape[0]
        device = scale_inputs[0][0].device
        proj_seqs, masks = [], []
        for i, (x, m) in enumerate(scale_inputs):
            h = self.projs[i](x)  # [B, T_i, d_model]
            pos = self.pos[i](h.shape[1], self._scale_tubelets[i], device=device)
            h = h + pos.unsqueeze(0) + self.stream_emb[i].view(1, 1, -1)
            proj_seqs.append(h)
            masks.append(m.bool())
        x = torch.cat(proj_seqs, dim=1)  # [B, sum(T_i), d_model]
        m = torch.cat(masks, dim=1)  # [B, sum(T_i)]

        cls = self.cls.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
        m = torch.cat([cls_mask, m], dim=1)
        key_padding_mask = ~m  # transformer expects True on padding to ignore
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h_cls = self.norm(x[:, 0])
        logits = self.head(h_cls)
        if self.aux_head is not None:
            aux = self.aux_head(h_cls)
            return logits, aux
        return logits


# ---------------------------------------------------------------------------
# CORN loss and decoding
# ---------------------------------------------------------------------------


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_levels: int = NUM_CLASSES,
              pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    # logits: [B, num_levels-1]; labels: [B] in [0, num_levels-1]
    B = logits.shape[0]
    levels = torch.arange(num_levels - 1, device=logits.device).unsqueeze(0)  # [1, K-1]
    targets = (labels.unsqueeze(1) > levels).float()  # [B, K-1]
    # Conditional training mask: level k contributes only when label > k-1, i.e. sample reached k
    # corn_loss formulation: only use loss on level k for samples with label >= k
    keep = (labels.unsqueeze(1) >= levels).float()
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight.to(logits.device)
        )
    else:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * keep).sum() / keep.sum().clamp_min(1.0)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    # Convert logits → conditional probabilities → cumulative product → labels
    p_cond = torch.sigmoid(logits)
    p_cum = torch.cumprod(p_cond, dim=1)
    # predicted label = number of levels passed (probability > 0.5)
    return (p_cum > 0.5).sum(dim=1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------


def class_weight_vec(y: np.ndarray, mode: str, manual: str = "") -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "manual":
        parts = [float(v) for v in manual.split(",") if v.strip()]
        if len(parts) != NUM_CLASSES:
            raise ValueError("--manual-class-weights must have NUM_CLASSES values")
        return torch.tensor(parts, dtype=torch.float32)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced":
        w = counts.sum() / (NUM_CLASSES * counts)
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
        w = w * (counts.sum() / (w * counts).sum())
    else:
        raise ValueError(mode)
    return torch.tensor(w, dtype=torch.float32)


def corn_pos_weight_from_labels(y: np.ndarray, max_ratio: float = 4.0) -> torch.Tensor:
    """Proper CORN pos_weight: at level k, weight = neg_count / pos_count.

    pos_count[k] = num samples with label > k (those whose level-k logit should be high)
    neg_count[k] = num samples with label <= k

    CORN only trains level k on samples with label >= k (conditional), so the
    relevant pos/neg counts are restricted to that conditional subset.
    Clipped to max_ratio to keep optimization stable; falls back to 1.0 when
    the conditional set is empty (e.g., class 4 absent from train fold).
    """
    K = NUM_CLASSES - 1
    pos = torch.ones(K, dtype=torch.float32)
    for k in range(K):
        in_subset = (y >= k)
        if not in_subset.any():
            continue
        pos_count = int(((y > k) & in_subset).sum())
        neg_count = int(((y == k) & in_subset).sum())
        if pos_count == 0 or neg_count == 0:
            continue
        pos[k] = float(neg_count) / float(pos_count)
    pos = pos.clamp(1.0 / max_ratio, max_ratio)
    return pos


def evaluate(model, scales, y, batch_size, device) -> dict:
    model.eval()
    preds = []
    probs_list = []
    n = len(y)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_inputs = []
            for s in scales:
                xs = torch.from_numpy(s["x"][i:i + batch_size]).to(device, dtype=torch.float32)
                ms = torch.from_numpy(s["mask"][i:i + batch_size]).to(device, dtype=torch.bool)
                batch_inputs.append((xs, ms))
            out = model(batch_inputs)
            logits = out[0] if isinstance(out, tuple) else out
            # CORN: per-level binary logits → conditional sigmoid → cumulative → class probs
            p_cond = torch.sigmoid(logits)
            p_cum = torch.cumprod(p_cond, dim=1)  # [B, K-1] = P(y > k) for k=0..K-2
            # Class probabilities: P(y=k) = P(y > k-1) - P(y > k); endpoints special
            B = logits.shape[0]
            K = NUM_CLASSES
            p_class = torch.zeros(B, K, device=logits.device)
            # P(y == 0) = 1 - P(y > 0)
            p_class[:, 0] = 1.0 - p_cum[:, 0]
            for k in range(1, K - 1):
                p_class[:, k] = p_cum[:, k - 1] - p_cum[:, k]
            p_class[:, K - 1] = p_cum[:, K - 2]
            p_class = p_class.clamp(min=0.0)
            p_class = p_class / p_class.sum(dim=1, keepdim=True).clamp_min(1e-8)
            yhat = corn_predict(logits).cpu().numpy()
            preds.append(yhat)
            probs_list.append(p_class.cpu().numpy())
    yhat = np.concatenate(preds, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    return {
        "acc": float(accuracy_score(y, yhat)),
        "qwk": float(cohen_kappa_score(y, yhat, weights="quadratic")),
        "mae": float(mean_absolute_error(y, yhat)),
        "preds": yhat.tolist(),
        "val_probs": probs.tolist(),
        "val_labels": y.tolist(),
    }


def cosine_warmup(epoch, warmup_epochs, cosine_epochs, base_lr, final_lr):
    """Warmup linearly to base_lr, then cosine-decay to final_lr over cosine_epochs.
    After cosine_epochs (post-warmup), lr stays at final_lr.
    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(1, warmup_epochs)
    rel = epoch - warmup_epochs
    if rel >= cosine_epochs:
        return final_lr
    t = rel / max(1, cosine_epochs)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not args.scale_names:
        args.scale_names = [None] * len(args.train_npz)
    if not args.tubelets_per_segment:
        args.tubelets_per_segment = [None] * len(args.train_npz)
    if not args.num_segments_for_pos:
        args.num_segments_for_pos = [None] * len(args.train_npz)
    if not (len(args.train_npz) == len(args.val_npz) == len(args.scale_names)
            == len(args.tubelets_per_segment) == len(args.num_segments_for_pos)):
        raise ValueError("All scale args must have the same count.")

    scales: list[ScaleCache] = []
    y_train_ref = None
    y_val_ref = None
    for tp, vp, nm, ts, ns in zip(
        args.train_npz, args.val_npz, args.scale_names,
        args.tubelets_per_segment, args.num_segments_for_pos,
    ):
        sc, y_tr, y_va = load_scale(tp, vp, nm, ns, ts)
        scales.append(sc)
        if y_train_ref is None:
            y_train_ref = y_tr
            y_val_ref = y_va
        else:
            if not np.array_equal(y_train_ref, y_tr):
                raise ValueError(f"train labels mismatch between scales: {tp}")
            if not np.array_equal(y_val_ref, y_va):
                raise ValueError(f"val labels mismatch between scales: {vp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[combiner] device={device} n_scales={len(scales)}", flush=True)
    for s in scales:
        print(f"  scale {s.name}: train={s.x_train.shape}, val={s.x_val.shape}, "
              f"num_seg={s.num_segments_for_pos}, tubelets/seg={s.tubelets_per_segment}",
              flush=True)

    # Load kinematic-distillation targets if requested
    aux_train = None
    aux_val = None
    aux_train_mask = None
    aux_val_mask = None
    aux_dim = 0
    feature_cols: list[str] = []
    if args.distill_train_csv and args.distill_val_csv:
        feature_cols = [c.strip() for c in args.distill_features.split(",") if c.strip()]
        if not (args.distill_fold_train_csv and args.distill_fold_val_csv):
            raise ValueError("--distill-fold-train-csv and --distill-fold-val-csv are required when using distillation.")
        # Parse coverage from the FIRST scale's NPZ (assumed shared across scales for ordering)
        first_train_npz = np.load(args.train_npz[0], allow_pickle=False)
        first_val_npz = np.load(args.val_npz[0], allow_pickle=False)
        cov_tr = None
        cov_va = None
        if "coverage" in first_train_npz.files:
            cov_tr = [json.loads(str(s)) for s in first_train_npz["coverage"]]
        if "coverage" in first_val_npz.files:
            cov_va = [json.loads(str(s)) for s in first_val_npz["coverage"]]
        aux_train_raw, aux_train_mask = load_distillation_targets(
            args.distill_fold_train_csv, args.distill_train_csv, feature_cols, cov_tr,
        )
        aux_val_raw, aux_val_mask = load_distillation_targets(
            args.distill_fold_val_csv, args.distill_val_csv, feature_cols, cov_va,
        )
        aux_train, aux_val, aux_mu, aux_sd = standardize_aux(
            aux_train_raw, aux_train_mask, aux_val_raw, aux_val_mask,
        )
        aux_dim = aux_train.shape[1]
        print(f"[combiner] distillation: features={feature_cols}", flush=True)
        print(f"[combiner] aux_train shape={aux_train.shape}, valid rows={aux_train_mask.sum()}/{len(aux_train_mask)}", flush=True)
        print(f"[combiner] aux_val   shape={aux_val.shape},  valid rows={aux_val_mask.sum()}/{len(aux_val_mask)}", flush=True)

    model = TemporalCornCombiner(
        scales=scales,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        aux_dim=aux_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[combiner] params={n_params/1e6:.2f}M", flush=True)

    if args.class_weight == "none":
        pos_w = None
    else:
        pos_w = corn_pos_weight_from_labels(y_train_ref)
    if pos_w is not None:
        print(f"[combiner] CORN pos_weight={pos_w.numpy().round(3).tolist()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_wandb = bool(args.wandb_project)
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or out_dir.name,
                mode=args.wandb_mode,
                config=vars(args),
            )
        except Exception as exc:
            print(f"[combiner] wandb init failed: {exc}", flush=True)
            use_wandb = False

    train_scale_views = [
        {"x": s.x_train, "mask": s.mask_train} for s in scales
    ]
    val_scale_views = [
        {"x": s.x_val, "mask": s.mask_val} for s in scales
    ]

    best = {"qwk": -2.0, "acc": -1.0, "epoch": -1, "metrics": None}
    epochs_since_improve = 0
    n_train = len(y_train_ref)
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        cosine_epochs = args.cosine_epochs if args.cosine_epochs > 0 else max(1, args.epochs - args.warmup_epochs)
        lr = cosine_warmup(epoch, args.warmup_epochs, cosine_epochs, args.lr, args.lr * args.final_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = rng.permutation(n_train)
        total_loss, total = 0.0, 0
        for i in range(0, n_train, args.batch_size):
            idx = order[i:i + args.batch_size]
            batch_inputs = []
            for scale_i, s in enumerate(train_scale_views):
                xs = torch.from_numpy(s["x"][idx]).to(device, dtype=torch.float32)
                ms = torch.from_numpy(s["mask"][idx]).to(device, dtype=torch.bool)
                if args.token_dropout > 0:
                    drop = torch.rand_like(ms, dtype=torch.float32) < args.token_dropout
                    ms = ms & ~drop
                if args.segment_dropout > 0:
                    sc = scales[scale_i]
                    n_seg = sc.num_segments_for_pos
                    seg_drop = torch.rand(xs.shape[0], n_seg, device=device) < args.segment_dropout
                    # Expand to token mask
                    rep = sc.tubelets_per_segment
                    if seg_drop.shape[1] * rep < ms.shape[1]:
                        seg_drop = torch.nn.functional.pad(seg_drop, (0, ms.shape[1] - seg_drop.shape[1] * rep), value=False)
                    seg_drop_tok = seg_drop.repeat_interleave(rep, dim=1)[:, :ms.shape[1]]
                    ms = ms & ~seg_drop_tok
                batch_inputs.append((xs, ms))
            y = torch.from_numpy(y_train_ref[idx]).to(device, dtype=torch.long)

            # Aux distillation targets for this batch (if enabled)
            aux_y = None
            aux_m = None
            if aux_train is not None:
                aux_y = torch.from_numpy(aux_train[idx]).to(device, dtype=torch.float32)
                aux_m = torch.from_numpy(aux_train_mask[idx]).to(device, dtype=torch.float32)

            if args.mixup_alpha > 0 and len(idx) > 1:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(len(idx), device=device)
                batch_inputs = [
                    (lam * xs + (1 - lam) * xs[perm], ms & ms[perm])
                    for (xs, ms) in batch_inputs
                ]
                out = model(batch_inputs)
                if isinstance(out, tuple):
                    logits, aux_pred = out
                else:
                    logits, aux_pred = out, None
                loss = lam * corn_loss(logits, y, pos_weight=pos_w) \
                    + (1 - lam) * corn_loss(logits, y[perm], pos_weight=pos_w)
                if aux_pred is not None and aux_y is not None:
                    aux_y_mix = lam * aux_y + (1 - lam) * aux_y[perm]
                    aux_m_mix = aux_m * aux_m[perm]
                    sq = (aux_pred - aux_y_mix) ** 2
                    aux_loss = (sq.mean(dim=1) * aux_m_mix).sum() / aux_m_mix.sum().clamp_min(1.0)
                    loss = loss + args.distill_weight * aux_loss
            else:
                out = model(batch_inputs)
                if isinstance(out, tuple):
                    logits, aux_pred = out
                else:
                    logits, aux_pred = out, None
                loss = corn_loss(logits, y, pos_weight=pos_w)
                if aux_pred is not None and aux_y is not None:
                    sq = (aux_pred - aux_y) ** 2
                    aux_loss = (sq.mean(dim=1) * aux_m).sum() / aux_m.sum().clamp_min(1.0)
                    loss = loss + args.distill_weight * aux_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(idx)
            total += len(idx)
        train_loss = total_loss / max(1, total)

        val_metrics = evaluate(model, val_scale_views, y_val_ref, args.batch_size, device)
        msg = (
            f"epoch={epoch:03d} lr={lr:.2e} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} qwk={val_metrics['qwk']:.4f} mae={val_metrics['mae']:.4f}"
        )
        print(msg, flush=True)
        if use_wandb:
            try:
                wandb.log({
                    "train/loss": train_loss,
                    "val/acc": val_metrics["acc"],
                    "val/qwk": val_metrics["qwk"],
                    "val/mae": val_metrics["mae"],
                    "lr": lr,
                    "epoch": epoch,
                })
            except Exception:
                pass

        key = (val_metrics["qwk"], val_metrics["acc"]) if args.select_by in ("qwk", "qwk_acc") else (val_metrics["acc"], val_metrics["qwk"])
        cur_best = (best["qwk"], best["acc"]) if args.select_by in ("qwk", "qwk_acc") else (best["acc"], best["qwk"])
        if key > cur_best:
            best.update({
                "qwk": val_metrics["qwk"],
                "acc": val_metrics["acc"],
                "epoch": epoch,
                "metrics": val_metrics,
            })
            epochs_since_improve = 0
            if args.save_best:
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch},
                           out_dir / "best.pt")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"[combiner] early stop at epoch={epoch}; best_qwk={best['qwk']:.4f}", flush=True)
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "best": best,
        "args": vars(args),
        "n_params": n_params,
        "n_train": int(n_train),
        "n_val": int(len(y_val_ref)),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result["best"].items() if k != "metrics"}, indent=2))
    if use_wandb and wandb_run is not None:
        try:
            wandb_run.summary["best_qwk"] = best["qwk"]
            wandb_run.summary["best_acc"] = best["acc"]
            wandb_run.summary["best_epoch"] = best["epoch"]
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
