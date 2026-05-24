"""CORN head with attention-based temporal aggregation across multi-segment clips.

Meta's default pipeline produces K logits per video (one per segment) and computes
the loss against the SAME video label for each segment — i.e. every 1-second
sub-clip is told to predict the full video's severity. For ordinal severity
grading this is the wrong inductive bias: adjacent grades (1↔2↔3) are
indistinguishable in many 1-second windows, so the model collapses toward the
most-frequent class.

This head pools across segments BEFORE the loss:
  1. Apply attentive pooler to each segment → (B, embed_dim) per segment.
  2. Stack into (B, K, embed_dim).
  3. Attention-pool over K with a learnable [CLS] query → (B, embed_dim).
  4. Single linear → CORN K-1 logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


class TemporalAggCORNHead(nn.Module):
    """Per-segment attentive pool → temporal attention pool → CORN logits."""

    # Marks this head as "needs the full list of K segment outputs",
    # consumed by the eval forward path.
    consumes_segment_list = True
    supports_segment_mask = True

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        depth: int = 1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        qkv_bias: bool = True,
        num_levels: int = 5,
        complete_block: bool = True,
        use_activation_checkpointing: bool = False,
        temporal_heads: int = 4,
        temporal_dropout: float = 0.1,
        include_length_features: bool = False,
        length_feature_scale: float = 0.1,
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
        # Temporal aggregator: learnable [CLS] attends over K segment features.
        self.temp_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.temp_cls, std=init_std)
        self.temp_attn = nn.MultiheadAttention(
            embed_dim, num_heads=temporal_heads, dropout=temporal_dropout, batch_first=True
        )
        self.temp_norm = nn.LayerNorm(embed_dim)
        self.temp_dropout = nn.Dropout(temporal_dropout)
        self.include_length_features = bool(include_length_features)
        self.length_feature_scale = float(length_feature_scale)
        length_dim = 3 if self.include_length_features else 0
        self.linear = nn.Linear(embed_dim + length_dim, self.num_levels - 1, bias=True)

    def pool_segment(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a single segment's tokens to (B, embed_dim)."""
        return self.pooler(x).squeeze(1)

    def aggregate(self, segment_feats: torch.Tensor, segment_mask: torch.Tensor | None = None) -> torch.Tensor:
        """segment_feats: (B, K, embed_dim) -> (B, num_levels - 1) CORN logits."""
        B = segment_feats.size(0)
        cls = self.temp_cls.expand(B, -1, -1)  # (B, 1, D)
        key_padding_mask = None
        if segment_mask is not None:
            segment_mask = segment_mask.to(device=segment_feats.device, dtype=torch.bool)
            key_padding_mask = ~segment_mask
        out, _ = self.temp_attn(
            cls,
            segment_feats,
            segment_feats,
            key_padding_mask=key_padding_mask,
        )
        out = self.temp_norm(out.squeeze(1))
        out = self.temp_dropout(out)
        if self.include_length_features:
            if segment_mask is None:
                segment_mask = torch.ones(
                    segment_feats.shape[:2],
                    dtype=torch.bool,
                    device=segment_feats.device,
                )
            out = torch.cat(
                [
                    out,
                    _length_features(
                        segment_mask,
                        dtype=out.dtype,
                        scale=self.length_feature_scale,
                    ),
                ],
                dim=1,
            )
        return self.linear(out)

    def forward(self, x, segment_mask: torch.Tensor | None = None):
        """Two call patterns:

        - `x` is a list/tuple of K tensors, each (B, N, D): full temporal-aggregate path.
        - `x` is a single tensor (B, N, D): falls back to per-segment pool + linear
          (no aggregation), kept so frozen-encoder eval still works on single inputs.
        """
        if isinstance(x, (list, tuple)):
            feats = torch.stack([self.pool_segment(seg) for seg in x], dim=1)
            return self.aggregate(feats, segment_mask=segment_mask)
        pooled = self.pool_segment(x)
        if self.include_length_features:
            mask = torch.ones((pooled.shape[0], 1), dtype=torch.bool, device=pooled.device)
            pooled = torch.cat(
                [
                    pooled,
                    _length_features(mask, dtype=pooled.dtype, scale=self.length_feature_scale),
                ],
                dim=1,
            )
        return self.linear(pooled)


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
