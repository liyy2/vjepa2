"""CORN ordinal attentive probe head."""

from __future__ import annotations

import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


class CORNAttentiveClassifier(nn.Module):
    """Meta attentive pooler with a K-1 CORN ordinal output layer."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        depth: int = 4,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        num_levels: int = 5,
        complete_block: bool = True,
        use_activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("CORN requires at least two ordinal levels.")
        self.num_levels = int(num_levels)
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.linear = nn.Linear(embed_dim, self.num_levels - 1, bias=True)

    def forward(self, x):
        x = self.pooler(x).squeeze(1)
        return self.linear(x)

