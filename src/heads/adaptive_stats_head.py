"""Mask-aware temporal statistics CORN head for adaptive V-JEPA windows."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveStatsCORNHead(nn.Module):
    """Pool variable-length per-window embeddings with masked statistics."""

    consumes_segment_list = True
    supports_segment_mask = True

    def __init__(
        self,
        embed_dim: int = 1024,
        num_levels: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.35,
        percentiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
        metadata_dim: int = 0,
        metadata_scale: float = 1.0,
        include_length_features: bool = False,
        length_feature_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("CORN requires at least two ordinal levels.")
        self.num_levels = int(num_levels)
        self.percentiles = tuple(float(p) for p in percentiles)
        self.metadata_dim = int(metadata_dim)
        self.metadata_scale = float(metadata_scale)
        self.include_length_features = bool(include_length_features)
        self.length_feature_scale = float(length_feature_scale)
        length_dim = 3 if self.include_length_features else 0
        feature_dim = int(embed_dim) * (4 + len(self.percentiles)) + self.metadata_dim + length_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.num_levels - 1),
        )

    def pool_segment(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().mean(dim=1)

    def forward(
        self,
        x,
        segment_mask: torch.Tensor | None = None,
        metadata_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(x, (list, tuple)):
            x = [x]
        seq = torch.stack([self.pool_segment(seg) for seg in x], dim=1)
        mask = _valid_mask(segment_mask, seq)
        features = _masked_stats(seq, mask, self.percentiles)
        if self.include_length_features:
            features.append(_length_features(mask, dtype=seq.dtype, scale=self.length_feature_scale))
        if self.metadata_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    seq.shape[0],
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


def _valid_mask(segment_mask: torch.Tensor | None, seq: torch.Tensor) -> torch.Tensor:
    if segment_mask is None:
        return torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
    mask = segment_mask.to(device=seq.device, dtype=torch.bool)
    if mask.shape != seq.shape[:2]:
        raise ValueError(f"segment_mask shape {tuple(mask.shape)} does not match {tuple(seq.shape[:2])}")
    return mask


def _masked_stats(
    seq: torch.Tensor,
    mask: torch.Tensor,
    percentiles: tuple[float, ...],
) -> list[torch.Tensor]:
    mask_f = mask.unsqueeze(-1).to(dtype=seq.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (seq * mask_f).sum(dim=1) / count
    centered = (seq - mean.unsqueeze(1)) * mask_f
    std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0.0))
    min_values = seq.masked_fill(~mask.unsqueeze(-1), torch.inf).min(dim=1).values
    max_values = seq.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    min_values = torch.where(torch.isfinite(min_values), min_values, mean)
    max_values = torch.where(torch.isfinite(max_values), max_values, mean)

    sorted_values = seq.masked_fill(~mask.unsqueeze(-1), torch.inf).sort(dim=1).values
    valid_count = mask.sum(dim=1).clamp_min(1)
    pct_values = []
    for pct in percentiles:
        pct = min(max(float(pct), 0.0), 1.0)
        idx = torch.floor((valid_count - 1).to(dtype=seq.dtype) * pct).long()
        idx = idx.view(seq.shape[0], 1, 1).expand(-1, 1, seq.shape[2])
        pct_values.append(sorted_values.gather(dim=1, index=idx).squeeze(1))
    return [mean, std, min_values, max_values, *pct_values]


def _length_features(mask: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    count = mask.sum(dim=1, keepdim=True).to(dtype=dtype)
    max_count = torch.full_like(count, max(mask.shape[1], 1))
    return torch.cat(
        [
            count * float(scale),
            torch.log1p(count),
            count / max_count,
        ],
        dim=1,
    )
