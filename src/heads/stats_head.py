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
    ) -> None:
        super().__init__()
        pooled_dim = 3 * int(embed_dim)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        max_values = x.max(dim=1).values
        return self.net(torch.cat((mean, std, max_values), dim=1))
