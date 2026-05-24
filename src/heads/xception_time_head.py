"""Mask-aware XceptionTime-style CORN head for adaptive V-JEPA windows."""

from __future__ import annotations

import torch
import torch.nn as nn


class XceptionTimeCORNHead(nn.Module):
    """Depthwise-separable 1D temporal conv stack over per-window embeddings."""

    consumes_segment_list = True
    supports_segment_mask = True

    def __init__(
        self,
        embed_dim: int = 1024,
        num_levels: int = 5,
        d_model: int = 128,
        hidden_dim: int = 512,
        depth: int = 4,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        dropout: float = 0.25,
        metadata_dim: int = 0,
        metadata_scale: float = 1.0,
        include_length_features: bool = False,
        length_feature_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("CORN requires at least two ordinal levels.")
        self.num_levels = int(num_levels)
        self.metadata_dim = int(metadata_dim)
        self.metadata_scale = float(metadata_scale)
        d_model = int(d_model)
        self.proj = nn.Sequential(
            nn.LayerNorm(int(embed_dim)),
            nn.Linear(int(embed_dim), d_model),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
        )
        kernels = tuple(int(k) for k in kernel_sizes)
        if not kernels:
            raise ValueError("kernel_sizes must contain at least one kernel")
        self.blocks = nn.ModuleList(
            [
                _SeparableTemporalBlock(
                    channels=d_model,
                    kernel_size=kernels[i % len(kernels)],
                    dropout=float(dropout),
                )
                for i in range(int(depth))
            ]
        )
        self.include_length_features = bool(include_length_features)
        self.length_feature_scale = float(length_feature_scale)
        length_dim = 3 if self.include_length_features else 0
        out_dim = 3 * d_model + self.metadata_dim + length_dim
        self.net = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, int(hidden_dim)),
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
        seq = self.proj(seq)
        mask_f = mask.unsqueeze(-1).to(dtype=seq.dtype)
        y = (seq * mask_f).transpose(1, 2)
        conv_mask = mask.unsqueeze(1).to(dtype=y.dtype)
        for block in self.blocks:
            y = block(y, conv_mask)
        y = y.transpose(1, 2)
        features = _masked_pool(y, mask)
        if self.include_length_features:
            features.append(_length_features(mask, dtype=y.dtype, scale=self.length_feature_scale))
        if self.metadata_dim > 0:
            if metadata_features is None:
                metadata_features = torch.zeros(
                    y.shape[0],
                    self.metadata_dim,
                    dtype=y.dtype,
                    device=y.device,
                )
            elif metadata_features.shape[1] != self.metadata_dim:
                raise ValueError(
                    f"metadata_features has width {metadata_features.shape[1]}, "
                    f"expected {self.metadata_dim}"
                )
            features.append(metadata_features.to(device=y.device, dtype=y.dtype) * self.metadata_scale)
        return self.net(torch.cat(features, dim=1))


class _SeparableTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x * mask)
        y = self.pointwise(y)
        y = self.norm(y)
        y = self.act(y)
        y = self.dropout(y)
        return (residual + y) * mask


def _valid_mask(segment_mask: torch.Tensor | None, seq: torch.Tensor) -> torch.Tensor:
    if segment_mask is None:
        return torch.ones(seq.shape[:2], dtype=torch.bool, device=seq.device)
    mask = segment_mask.to(device=seq.device, dtype=torch.bool)
    if mask.shape != seq.shape[:2]:
        raise ValueError(f"segment_mask shape {tuple(mask.shape)} does not match {tuple(seq.shape[:2])}")
    return mask


def _masked_pool(seq: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    mask_f = mask.unsqueeze(-1).to(dtype=seq.dtype)
    count = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (seq * mask_f).sum(dim=1) / count
    centered = (seq - mean.unsqueeze(1)) * mask_f
    std = torch.sqrt((centered.square().sum(dim=1) / count).clamp_min(0.0))
    max_values = seq.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    max_values = torch.where(torch.isfinite(max_values), max_values, mean)
    return [mean, std, max_values]


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
