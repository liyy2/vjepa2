"""Simple statistics-pooling probe heads for frozen V-JEPA features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class StatsPoolingClassifier(nn.Module):
    """Classify a token sequence from mean/std/max pooled frozen features."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        metadata_dim: int = 0,
        metadata_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.metadata_dim = int(metadata_dim)
        self.metadata_scale = float(metadata_scale)
        pooled_dim = 3 * int(embed_dim) + self.metadata_dim
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor, metadata_features: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        max_values = x.max(dim=1).values
        pooled = [mean, std, max_values]
        if self.metadata_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    x.shape[0],
                    self.metadata_dim,
                    dtype=x.dtype,
                    device=x.device,
                )
            elif metadata_features.shape[1] != self.metadata_dim:
                raise ValueError(
                    f"metadata_features has width {metadata_features.shape[1]}, "
                    f"expected {self.metadata_dim}"
                )
            pooled.append(metadata_features.to(device=x.device, dtype=x.dtype) * self.metadata_scale)
        return self.net(torch.cat(pooled, dim=1))


class TemporalStatsPoolingClassifier(nn.Module):
    """Classify spatially pooled V-JEPA tubelet sequences with temporal stats."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1000,
        spatial_tokens: int = 576,
        hidden_dim: int = 512,
        d_model: int = 128,
        dropout: float = 0.35,
        temporal_bins: int = 12,
        spatial_pool: str = "mean",
        feature_stats_npz: str | None = None,
        fit_pca_dim: int = 0,
        feature_eps: float = 1.0e-4,
        metadata_dim: int = 0,
        metadata_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial_tokens = int(spatial_tokens)
        self.temporal_bins = int(temporal_bins)
        self.spatial_pool = spatial_pool
        self.metadata_dim = int(metadata_dim)
        self.metadata_scale = float(metadata_scale)
        if self.spatial_tokens <= 0:
            raise ValueError("spatial_tokens must be positive")
        if self.temporal_bins <= 0:
            raise ValueError("temporal_bins must be positive")
        if self.spatial_pool not in ("mean", "mean_std"):
            raise ValueError("spatial_pool must be 'mean' or 'mean_std'")

        pooled_dim = int(embed_dim) * (2 if self.spatial_pool == "mean_std" else 1)
        feature_dim_in = pooled_dim
        feature_mean, feature_std, pca_components = self._fit_feature_transform(
            feature_stats_npz=feature_stats_npz,
            feature_dim=pooled_dim,
            pca_dim=int(fit_pca_dim or 0),
            eps=float(feature_eps),
        )
        self.register_buffer("feature_mean", feature_mean, persistent=False)
        self.register_buffer("feature_std", feature_std, persistent=False)
        self.register_buffer("pca_components", pca_components, persistent=False)
        if self.pca_components.numel() > 0:
            feature_dim_in = int(self.pca_components.shape[0])

        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim_in),
            nn.Linear(feature_dim_in, int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        feature_dim = int(d_model) * (6 + 2 * self.temporal_bins) + self.metadata_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), num_classes),
        )

    def forward(self, x: torch.Tensor, metadata_features: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        batch_size, num_tokens, embed_dim = x.shape
        if num_tokens % self.spatial_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} is not divisible by spatial_tokens={self.spatial_tokens}"
            )
        temporal_tokens = num_tokens // self.spatial_tokens
        x = x.reshape(batch_size, temporal_tokens, self.spatial_tokens, embed_dim)
        mean = x.mean(dim=2)
        if self.spatial_pool == "mean_std":
            std = x.std(dim=2, unbiased=False)
            seq = torch.cat([mean, std], dim=-1)
        else:
            seq = mean

        if self.feature_mean.numel() > 0:
            seq = (seq - self.feature_mean.to(device=seq.device, dtype=seq.dtype)) / self.feature_std.to(
                device=seq.device,
                dtype=seq.dtype,
            )
        if self.pca_components.numel() > 0:
            seq = torch.matmul(seq, self.pca_components.to(device=seq.device, dtype=seq.dtype).t())

        seq = self.proj(seq)
        features = [
            seq.mean(dim=1),
            seq.std(dim=1, unbiased=False),
            seq.max(dim=1).values,
            seq[:, 0],
            seq[:, -1],
            seq[:, -1] - seq[:, 0],
        ]
        for start, end in self._bin_ranges(temporal_tokens, seq.device):
            chunk = seq[:, start:end]
            features.append(chunk.mean(dim=1))
        for start, end in self._bin_ranges(temporal_tokens, seq.device):
            chunk = seq[:, start:end]
            features.append(chunk.std(dim=1, unbiased=False))

        if self.metadata_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    batch_size,
                    self.metadata_dim,
                    dtype=seq.dtype,
                    device=seq.device,
                )
            elif metadata_features.shape[1] != self.metadata_dim:
                raise ValueError(
                    f"metadata_features has width {metadata_features.shape[1]}, "
                    f"expected {self.metadata_dim}"
                )
            features.append(metadata_features.to(device=seq.device, dtype=seq.dtype) * self.metadata_scale)
        return self.net(torch.cat(features, dim=1))

    def _bin_ranges(self, temporal_tokens: int, device: torch.device) -> list[tuple[int, int]]:
        del device
        ranges = []
        for bin_idx in range(self.temporal_bins):
            start = int(bin_idx * temporal_tokens // self.temporal_bins)
            end = int((bin_idx + 1) * temporal_tokens // self.temporal_bins)
            if end <= start:
                end = min(temporal_tokens, start + 1)
            ranges.append((start, end))
        return ranges

    def _fit_feature_transform(
        self,
        feature_stats_npz: str | None,
        feature_dim: int,
        pca_dim: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not feature_stats_npz:
            return torch.empty(0), torch.empty(0), torch.empty(0)

        data = np.load(feature_stats_npz)
        x = data["x"].astype(np.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected cached feature tensor with shape N,T,D, got {x.shape}")
        if x.shape[-1] != feature_dim:
            raise ValueError(
                f"Cached feature dim {x.shape[-1]} does not match head feature dim {feature_dim}"
            )
        mean = x.mean(axis=(0, 1), keepdims=False)
        std = np.maximum(x.std(axis=(0, 1), keepdims=False), eps)
        pca_components = np.empty((0,), dtype=np.float32)
        if pca_dim > 0:
            if pca_dim >= feature_dim:
                raise ValueError(f"fit_pca_dim={pca_dim} must be smaller than feature_dim={feature_dim}")
            from sklearn.decomposition import PCA

            flat = ((x.reshape(-1, feature_dim) - mean) / std).astype(np.float32)
            pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0, whiten=False)
            pca.fit(flat)
            pca_components = pca.components_.astype(np.float32)
            print(
                "TemporalStatsPoolingClassifier PCA "
                f"{feature_dim}->{pca_dim} explained_variance={float(pca.explained_variance_ratio_.sum()):.5f}",
                flush=True,
            )
        return (
            torch.from_numpy(mean.astype(np.float32)).view(1, 1, feature_dim),
            torch.from_numpy(std.astype(np.float32)).view(1, 1, feature_dim),
            torch.from_numpy(pca_components),
        )


class TemporalTransformerClassifier(nn.Module):
    """Classify spatially pooled V-JEPA tubelet sequences with a small Transformer."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 1000,
        spatial_tokens: int = 576,
        hidden_dim: int = 512,
        d_model: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_temporal_tokens: int = 512,
        spatial_pool: str = "mean",
        pool: str = "cls_mean",
        feature_stats_npz: str | None = None,
        fit_pca_dim: int = 0,
        feature_eps: float = 1.0e-4,
        metadata_dim: int = 0,
        metadata_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.spatial_tokens = int(spatial_tokens)
        self.spatial_pool = spatial_pool
        self.pool = pool
        self.metadata_dim = int(metadata_dim)
        self.metadata_scale = float(metadata_scale)
        if self.spatial_tokens <= 0:
            raise ValueError("spatial_tokens must be positive")
        if self.spatial_pool not in ("mean", "mean_std"):
            raise ValueError("spatial_pool must be 'mean' or 'mean_std'")
        if self.pool not in ("cls", "mean", "cls_mean", "stats"):
            raise ValueError("pool must be 'cls', 'mean', 'cls_mean', or 'stats'")

        pooled_dim = int(embed_dim) * (2 if self.spatial_pool == "mean_std" else 1)
        feature_dim_in = pooled_dim
        feature_mean, feature_std, pca_components = TemporalStatsPoolingClassifier._fit_feature_transform(
            self,
            feature_stats_npz=feature_stats_npz,
            feature_dim=pooled_dim,
            pca_dim=int(fit_pca_dim or 0),
            eps=float(feature_eps),
        )
        self.register_buffer("feature_mean", feature_mean, persistent=False)
        self.register_buffer("feature_std", feature_std, persistent=False)
        self.register_buffer("pca_components", pca_components, persistent=False)
        if self.pca_components.numel() > 0:
            feature_dim_in = int(self.pca_components.shape[0])

        d_model = int(d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim_in),
            nn.Linear(feature_dim_in, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, int(max_temporal_tokens) + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))
        if self.pool == "cls":
            out_dim = d_model
        elif self.pool == "mean":
            out_dim = 2 * d_model
        elif self.pool == "cls_mean":
            out_dim = 3 * d_model
        else:
            out_dim = 6 * d_model
        out_dim += self.metadata_dim
        self.net = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, metadata_features: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        batch_size, num_tokens, embed_dim = x.shape
        if num_tokens % self.spatial_tokens != 0:
            raise ValueError(
                f"num_tokens={num_tokens} is not divisible by spatial_tokens={self.spatial_tokens}"
            )
        temporal_tokens = num_tokens // self.spatial_tokens
        if temporal_tokens + 1 > self.pos_embed.shape[1]:
            raise ValueError(
                f"temporal_tokens={temporal_tokens} exceeds max_temporal_tokens="
                f"{self.pos_embed.shape[1] - 1}"
            )
        x = x.reshape(batch_size, temporal_tokens, self.spatial_tokens, embed_dim)
        mean = x.mean(dim=2)
        if self.spatial_pool == "mean_std":
            std = x.std(dim=2, unbiased=False)
            seq = torch.cat([mean, std], dim=-1)
        else:
            seq = mean

        if self.feature_mean.numel() > 0:
            seq = (seq - self.feature_mean.to(device=seq.device, dtype=seq.dtype)) / self.feature_std.to(
                device=seq.device,
                dtype=seq.dtype,
            )
        if self.pca_components.numel() > 0:
            seq = torch.matmul(seq, self.pca_components.to(device=seq.device, dtype=seq.dtype).t())

        seq = self.proj(seq)
        cls = self.cls_token.to(dtype=seq.dtype).expand(batch_size, -1, -1)
        seq = torch.cat([cls, seq], dim=1)
        seq = seq + self.pos_embed[:, : seq.shape[1]].to(device=seq.device, dtype=seq.dtype)
        seq = self.encoder(seq)
        cls_out = seq[:, 0]
        tokens = seq[:, 1:]
        mean_out = tokens.mean(dim=1)
        std_out = tokens.std(dim=1, unbiased=False)
        if self.pool == "cls":
            features = [cls_out]
        elif self.pool == "mean":
            features = [mean_out, std_out]
        elif self.pool == "cls_mean":
            features = [cls_out, mean_out, std_out]
        else:
            features = [
                mean_out,
                std_out,
                tokens.max(dim=1).values,
                tokens[:, 0],
                tokens[:, -1],
                tokens[:, -1] - tokens[:, 0],
            ]

        if self.metadata_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    batch_size,
                    self.metadata_dim,
                    dtype=seq.dtype,
                    device=seq.device,
                )
            elif metadata_features.shape[1] != self.metadata_dim:
                raise ValueError(
                    f"metadata_features has width {metadata_features.shape[1]}, "
                    f"expected {self.metadata_dim}"
                )
            features.append(metadata_features.to(device=seq.device, dtype=seq.dtype) * self.metadata_scale)
        return self.net(torch.cat(features, dim=1))
