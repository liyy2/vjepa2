#!/usr/bin/env python
"""Train small temporal encoders on cached pure-vision V-JEPA 2.1 embeddings."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_EMB = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0"
)
DEFAULT_OUT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/fold_0"
)


@dataclass(frozen=True)
class TrainConfig:
    model_type: str
    seed: int
    lr: float
    weight_decay: float
    dropout: float
    hidden_dim: int = 512
    d_model: int = 128
    layers: int = 2
    temporal_bins: int = 12
    class_weight: str = "balanced"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default=f"{DEFAULT_EMB}/vjepa21_vitl384_train_32f_step1_12seg_handcrop_mean.npz")
    parser.add_argument("--val-npz", default=f"{DEFAULT_EMB}/vjepa21_vitl384_val_32f_step1_12seg_handcrop_mean.npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-configs", type=int, default=0, help="0 means run the default full grid.")
    parser.add_argument("--start-config", type=int, default=1, help="1-indexed inclusive config start.")
    parser.add_argument("--end-config", type=int, default=0, help="1-indexed inclusive config end; 0 means last.")
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--pca-dim", type=int, default=0, help="Fit PCA on train tubelet features before temporal heads.")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_train, y_train = load_npz(args.train_npz)
    x_val, y_val = load_npz(args.val_npz)
    raw_train_shape = x_train.shape
    raw_val_shape = x_val.shape
    x_train, x_val = normalize_from_train(x_train, x_val)
    if args.pca_dim > 0:
        x_train, x_val = temporal_pca_from_train(x_train, x_val, args.pca_dim)

    all_configs = default_grid()
    start_config = max(1, args.start_config)
    end_config = args.end_config if args.end_config > 0 else len(all_configs)
    configs = all_configs[start_config - 1 : end_config]
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    wandb_run = maybe_init_wandb(args, x_train.shape, x_val.shape, len(configs))
    results = []
    best: dict[str, Any] | None = None
    partial_path = out_dir / "temporal_encoder_partial_results.json"
    for local_idx, cfg in enumerate(configs, start=1):
        idx = start_config + local_idx - 1
        print(f"config {idx}/{len(all_configs)} {cfg}", flush=True)
        result = train_one_config(
            cfg=cfg,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            patience=args.patience,
        )
        result["config_index"] = idx
        results.append(result)
        if best is None or _rank_key(result) > _rank_key(best):
            best = result
            print(
                "new_best "
                f"{best['correct']}/{len(y_val)} acc={best['val_acc']:.5f} "
                f"qwk={best['qwk']:.5f} mae={best['mae']:.5f} cfg={best['config']}",
                flush=True,
            )
        partial = sorted(results, key=_rank_key, reverse=True)
        partial_path.write_text(json.dumps({"best": partial[0], "top_results": partial[:50]}, indent=2))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "config_index": idx,
                    "config_val_acc": result["val_acc"],
                    "config_correct": result["correct"],
                    "config_qwk": result["qwk"],
                    "config_mae": result["mae"],
                    "best_val_acc": partial[0]["val_acc"],
                    "best_correct": partial[0]["correct"],
                    "best_qwk": partial[0]["qwk"],
                    "best_mae": partial[0]["mae"],
                }
            )

    results = sorted(results, key=_rank_key, reverse=True)
    ensembles = build_topk_ensembles(results, y_val)
    overall = sorted(results + ensembles, key=_rank_key, reverse=True)
    output = {
        "note": "Pure-vision temporal encoders trained on cached V-JEPA 2.1 spatially pooled tubelet embeddings.",
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "train_shape": list(x_train.shape),
        "val_shape": list(x_val.shape),
        "raw_train_shape": list(raw_train_shape),
        "raw_val_shape": list(raw_val_shape),
        "pca_dim": args.pca_dim,
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
        f"BEST {overall[0]['correct']}/{len(y_val)} acc={overall[0]['val_acc']:.5f} "
        f"qwk={overall[0]['qwk']:.5f} mae={overall[0]['mae']:.5f}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "final_best_val_acc": overall[0]["val_acc"],
                "final_best_correct": overall[0]["correct"],
                "final_best_qwk": overall[0]["qwk"],
                "final_best_mae": overall[0]["mae"],
            }
        )
        wandb_run.finish()


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    return x, y


def normalize_from_train(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-4)
    return (x_train - mean) / std, (x_val - mean) / std


def temporal_pca_from_train(x_train: np.ndarray, x_val: np.ndarray, pca_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if pca_dim >= x_train.shape[-1]:
        return x_train, x_val
    n_train, t_train, d_train = x_train.shape
    n_val, t_val, d_val = x_val.shape
    if d_train != d_val:
        raise ValueError(f"train/val embedding dims differ: {d_train} vs {d_val}")
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0, whiten=False)
    train_flat = x_train.reshape(-1, d_train)
    val_flat = x_val.reshape(-1, d_val)
    train_reduced = pca.fit_transform(train_flat).astype(np.float32)
    val_reduced = pca.transform(val_flat).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"fit temporal PCA: {d_train} -> {pca_dim}, explained_variance={explained:.5f}", flush=True)
    return train_reduced.reshape(n_train, t_train, pca_dim), val_reduced.reshape(n_val, t_val, pca_dim)


def default_grid() -> list[TrainConfig]:
    configs: list[TrainConfig] = []
    seeds = [0, 17, 42, 98]
    for class_weight in ["balanced", "none"]:
        for seed in seeds:
            for lr in [1e-3, 3e-4]:
                for wd in [1e-2, 1e-1]:
                    configs.append(
                        TrainConfig(
                            "stats_mlp",
                            seed,
                            lr,
                            wd,
                            0.35,
                            hidden_dim=512,
                            temporal_bins=12,
                            class_weight=class_weight,
                        )
                    )
                    configs.append(
                        TrainConfig(
                            "stats_mlp",
                            seed,
                            lr,
                            wd,
                            0.5,
                            hidden_dim=256,
                            temporal_bins=24,
                            class_weight=class_weight,
                        )
                    )
            for lr in [1e-3, 3e-4]:
                configs.append(TrainConfig("gru", seed, lr, 1e-2, 0.3, d_model=128, layers=1, class_weight=class_weight))
                configs.append(
                    TrainConfig(
                        "transformer",
                        seed,
                        lr,
                        1e-2,
                        0.3,
                        d_model=128,
                        layers=2,
                        class_weight=class_weight,
                    )
                )
    return configs


def train_one_config(
    cfg: TrainConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    device: torch.device,
    patience: int,
) -> dict[str, Any]:
    set_seed(cfg.seed)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)

    model = build_model(cfg, input_dim=x_train.shape[-1], num_classes=5, seq_len=x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = class_weights(y_train, mode=cfg.class_weight)
    if weights is not None:
        weights = weights.to(device)
    x_val_t = torch.from_numpy(x_val).to(device)

    best: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            logits = model(x_val_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        result = metrics_from_probs(y_val, probs)
        result["epoch"] = epoch
        if best is None or _rank_key(result) > _rank_key(best):
            best = result
            best["val_probs"] = probs.tolist()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    assert best is not None
    best["config"] = asdict(cfg)
    best["epochs_ran"] = epoch
    return best


def build_model(cfg: TrainConfig, input_dim: int, num_classes: int, seq_len: int) -> nn.Module:
    if cfg.model_type == "stats_mlp":
        return TemporalStatsMLP(input_dim, num_classes, cfg.hidden_dim, cfg.dropout, cfg.temporal_bins, cfg.d_model)
    if cfg.model_type == "gru":
        return TemporalGRU(input_dim, num_classes, cfg.d_model, cfg.layers, cfg.dropout)
    if cfg.model_type == "transformer":
        return TemporalTransformer(input_dim, num_classes, cfg.d_model, cfg.layers, cfg.dropout, seq_len)
    raise ValueError(f"Unknown model_type={cfg.model_type}")


class TemporalStatsMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        temporal_bins: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.temporal_bins = int(temporal_bins)
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        feature_dim = d_model * (6 + 2 * self.temporal_bins)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        chunks = torch.chunk(x, self.temporal_bins, dim=1)
        bin_mean = [chunk.mean(dim=1) for chunk in chunks]
        bin_std = [chunk.std(dim=1, unbiased=False) for chunk in chunks]
        features = [
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
            x.max(dim=1).values,
            x[:, 0],
            x[:, -1],
            x[:, -1] - x[:, 0],
            *bin_mean,
            *bin_std,
        ]
        return self.net(torch.cat(features, dim=1))


class TemporalGRU(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model))
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        out, _ = self.gru(x)
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(pooled)


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int, layers: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.pos_embed[:, : x.shape[1]]
        x = self.encoder(x)
        attn = torch.softmax(torch.matmul(self.query.expand(x.shape[0], -1, -1), x.transpose(1, 2)), dim=-1)
        pooled = torch.matmul(attn, x).squeeze(1)
        return self.head(pooled)


def class_weights(y: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode != "balanced":
        return None
    counts = np.bincount(y, minlength=5).astype(np.float32)
    total = counts.sum()
    weights = np.zeros_like(counts)
    nonzero = counts > 0
    weights[nonzero] = total / (5.0 * counts[nonzero])
    return torch.from_numpy(weights)


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1).astype(np.int64)
    return metrics(labels, pred)


def metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "correct": int((pred == labels).sum()),
        "val_acc": float(accuracy_score(labels, pred)),
        "qwk": float(cohen_kappa_score(labels, pred, weights="quadratic")),
        "mae": float(mean_absolute_error(labels, pred)),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1, 2, 3, 4]).tolist(),
        "predictions": pred.tolist(),
        "labels": labels.tolist(),
    }


def build_topk_ensembles(results: list[dict[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    ensembles = []
    probs = [np.asarray(result["val_probs"], dtype=np.float32) for result in results if "val_probs" in result]
    if len(probs) < 2:
        return ensembles
    running = np.zeros_like(probs[0])
    for idx, prob in enumerate(probs[: min(30, len(probs))], start=1):
        running += prob
        if idx < 2:
            continue
        result = metrics_from_probs(labels, running / idx)
        result["ensemble_top_k"] = idx
        result["config"] = {"model_type": "topk_prob_ensemble", "top_k": idx}
        ensembles.append(result)
    return sorted(ensembles, key=_rank_key, reverse=True)


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float]:
    return (float(result["correct"]), float(result["qwk"]), -float(result["mae"]))


def maybe_init_wandb(args: argparse.Namespace, train_shape: tuple[int, ...], val_shape: tuple[int, ...], num_configs: int) -> Any:
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging", flush=True)
        return None
    try:
        return wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            mode=args.wandb_mode,
            config={
                "train_npz": args.train_npz,
                "val_npz": args.val_npz,
                "train_shape": train_shape,
                "val_shape": val_shape,
                "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_configs": num_configs,
            "start_config": args.start_config,
            "end_config": args.end_config,
            "patience": args.patience,
            "pca_dim": args.pca_dim,
        },
    )
    except Exception as exc:
        print(f"wandb init failed; continuing without W&B logging: {type(exc).__name__}: {exc}", flush=True)
        return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
