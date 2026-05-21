#!/usr/bin/env python
"""Single strong Transformer classifier over V-JEPA spatiotemporal tokens.

One architecture, sensible defaults, longer training with proper LR schedule.
No sweep, no ensembling — per the goal of avoiding over-engineering.

  Input  : [N, T=192, D] PCA-128 features
  Pool   : reshape into [N, S=12 segments, T_per_seg, D] then mean over per_seg → [N, S, D]
  Head   : prepend [CLS], add learned segment positions, K Transformer layers,
           take [CLS] → linear → 5 classes
  Loss   : cross-entropy with label smoothing
  Sched  : cosine LR with warmup
  Train  : long-horizon with early-stop on val QWK

Usage:
  python scripts/pd_hand/train_temporal_transformer.py \\
      --train-npz <handcrop+nocrop concat .npz> \\
      --val-npz   <val .npz> \\
      --out-dir   <results dir>
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", required=True)
    p.add_argument("--val-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--num-segments", type=int, default=12,
                   help="reshape 192 tokens as S × (192/S). Matches the multi-clip eval grid.")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class-weight", choices=["none", "balanced", "sqrt"], default="balanced")
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    return p.parse_args()


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@dataclass
class Batch:
    x: torch.Tensor  # [B, S, D]
    y: torch.Tensor  # [B]


class TemporalTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int, n_heads: int, n_layers: int,
                 ffn_mult: int, dropout: float, num_classes: int, num_segments: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, num_segments + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_mult * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, S, D]
        B = x.shape[0]
        x = self.proj(x)
        cls = self.cls.expand(B, -1, -1)  # [B, 1, d]
        x = torch.cat([cls, x], dim=1)  # [B, S+1, d]
        x = x + self.pos
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # take CLS
        return self.head(x)


def load_npz(path):
    d = np.load(path)
    return d["x"], d["y"].astype(np.int64)


def to_segments(x, num_segments):
    """[N, T, D] -> [N, S, D] with mean pool inside each segment."""
    N, T, D = x.shape
    assert T % num_segments == 0, f"T={T} not divisible by S={num_segments}"
    per = T // num_segments
    x = x.reshape(N, num_segments, per, D).mean(axis=2)
    return x


def fit_pca(x, dim):
    """Fit PCA on flat [N*T, D] features, return [W, mu] for transform."""
    N, S, D = x.shape
    flat = x.reshape(-1, D).astype(np.float32)
    mu = flat.mean(axis=0)
    centered = flat - mu
    # SVD on (limited) — use randomized for large D
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=0)
    pca.fit(centered)
    W = pca.components_.astype(np.float32)  # [dim, D]
    return W, mu, pca.explained_variance_ratio_.sum()


def apply_pca(x, W, mu):
    N, S, D = x.shape
    flat = x.reshape(-1, D).astype(np.float32) - mu
    out = flat @ W.T
    return out.reshape(N, S, -1)


def cosine_lr(step, warmup, total, base_lr, final_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * t))


def qwk(y, p, n_cls=5):
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


def mae(y, p):
    return float(np.mean(np.abs(np.array(y) - np.array(p))))


def class_weights(y, mode):
    counts = np.bincount(y, minlength=5).astype(np.float32)
    counts = np.clip(counts, 1, None)
    if mode == "balanced":
        w = 1.0 / counts
    elif mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
    else:
        w = np.ones(5, dtype=np.float32)
    return (w * len(w) / w.sum()).astype(np.float32)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"loading {args.train_npz}")
    xtr, ytr = load_npz(args.train_npz)
    xv, yv = load_npz(args.val_npz)
    print(f"  train x={xtr.shape} y dist={dict(zip(*np.unique(ytr, return_counts=True)))}")
    print(f"  val   x={xv.shape}   y dist={dict(zip(*np.unique(yv, return_counts=True)))}")

    # Segment-pool
    xtr = to_segments(xtr, args.num_segments)
    xv = to_segments(xv, args.num_segments)
    print(f"  segment-pool: train={xtr.shape} val={xv.shape}")

    # PCA on train, apply to both
    W, mu, evr = fit_pca(xtr, args.pca_dim)
    xtr = apply_pca(xtr, W, mu)
    xv = apply_pca(xv, W, mu)
    print(f"  PCA: dim={args.pca_dim} explained_var={evr:.3f}")

    # To tensors
    xtr_t = torch.from_numpy(xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    xv_t = torch.from_numpy(xv).float().to(device)
    yv_t = torch.from_numpy(yv).long().to(device)
    cw = torch.from_numpy(class_weights(ytr, args.class_weight)).to(device)
    print(f"  class weights: {cw.cpu().numpy().round(3)}")

    model = TemporalTransformer(
        d_in=args.pca_dim, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_mult=args.ffn_mult, dropout=args.dropout,
        num_classes=args.num_classes, num_segments=args.num_segments,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params/1e6:.2f}M params (d_model={args.d_model}, layers={args.n_layers})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    N = xtr.shape[0]
    steps_per_epoch = max(1, (N + args.batch_size - 1) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    best = {"qwk": -1, "epoch": -1, "acc": 0.0, "mae": 9, "preds": None, "probs": None}
    history = []
    bad = 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        losses = []
        for bstart in range(0, N, args.batch_size):
            idx = perm[bstart:bstart + args.batch_size]
            xb = xtr_t[idx]; yb = ytr_t[idx]
            # Mixup at segment level (just on the input)
            if args.mixup_alpha > 0 and torch.rand(1).item() < 0.5:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                ridx = idx[torch.randperm(len(idx), device=device)]
                xb = lam * xb + (1 - lam) * xtr_t[ridx]
                yr = ytr_t[ridx]
            else:
                lam = 1.0
                yr = yb
            # Step
            step_i = epoch * steps_per_epoch + (bstart // args.batch_size)
            lr_now = cosine_lr(step_i, args.warmup * steps_per_epoch, total_steps,
                              args.lr, args.lr * 0.01)
            for g in optim.param_groups:
                g["lr"] = lr_now
            optim.zero_grad()
            logits = model(xb)
            loss = lam * F.cross_entropy(logits, yb, weight=cw, label_smoothing=args.label_smoothing) + \
                   (1 - lam) * F.cross_entropy(logits, yr, weight=cw, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

        # Eval
        model.eval()
        with torch.no_grad():
            logits = model(xv_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
        acc = float((preds == yv).mean())
        q = qwk(yv, preds)
        m = mae(yv, preds)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": acc, "qwk": q, "mae": m, "lr": lr_now})
        improved = q > best["qwk"]
        if improved:
            best = {"qwk": q, "epoch": epoch, "acc": acc, "mae": m, "preds": preds.tolist(), "probs": probs.tolist()}
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or improved:
            print(f"  ep{epoch:3d}  loss={np.mean(losses):.4f}  acc={acc:.4f} qwk={q:.4f} mae={m:.3f}  lr={lr_now:.2e}{' ★' if improved else ''}")
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch} (no improvement for {bad} epochs)")
            break

    print(f"\nBEST: epoch={best['epoch']}  qwk={best['qwk']:.4f}  acc={best['acc']:.4f}  mae={best['mae']:.3f}")
    results = {
        "best": best,
        "config": vars(args),
        "history": history,
        "explained_var": float(evr),
        "n_params": int(n_params),
        "n_train": int(N), "n_val": int(len(yv)),
    }
    (out / "transformer_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out / 'transformer_results.json'}")


if __name__ == "__main__":
    main()
