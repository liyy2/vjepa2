"""Simple statistics-pooling probe heads for frozen V-JEPA features."""

from __future__ import annotations

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
