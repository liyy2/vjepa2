"""Window-level multiple-instance CORN head for adaptive clip sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attentive_pooler import AttentivePooler


class WindowMILCORNHead(nn.Module):
    """Apply a CORN attentive scorer to each window, then aggregate windows.

    The parameter names intentionally match ``CORNAttentiveClassifier``
    (``pooler`` and ``linear``), so a fixed-grid CORN checkpoint can initialize
    this head directly. The head consumes a list of per-window token tensors and
    returns one clip-level CORN logit vector.
    """

    consumes_segment_list = True
    supports_segment_mask = True

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
        aggregation: str = "mean_prob",
        topk: int = 4,
        freeze_scorer: bool = False,
        stats_hidden_dim: int = 128,
        stats_dropout: float = 0.25,
        stats_percentiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
        stats_topk: tuple[int, ...] = (1, 2, 4, 8),
        stats_source: str = "prob_logit",
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("CORN requires at least two ordinal levels.")
        self.num_levels = int(num_levels)
        self.aggregation = str(aggregation)
        self.topk = int(topk)
        self.stats_percentiles = tuple(float(p) for p in stats_percentiles)
        self.stats_topk = tuple(int(k) for k in stats_topk)
        self.stats_source = str(stats_source)
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
        if freeze_scorer:
            for param in self.pooler.parameters():
                param.requires_grad = False
            for param in self.linear.parameters():
                param.requires_grad = False

        if self.aggregation in {"learned_stats", "stats_mlp"}:
            stats_dim = _stats_feature_dim(
                num_logits=self.num_levels - 1,
                percentiles=self.stats_percentiles,
                topks=self.stats_topk,
                source=self.stats_source,
            )
            self.stats_mlp = nn.Sequential(
                nn.LayerNorm(stats_dim),
                nn.Linear(stats_dim, int(stats_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(stats_dropout)),
                nn.Linear(int(stats_hidden_dim), int(stats_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(stats_dropout)),
                nn.Linear(int(stats_hidden_dim), self.num_levels - 1),
            )
        else:
            self.stats_mlp = None

    def forward(self, x, segment_mask: torch.Tensor | None = None):
        if isinstance(x, torch.Tensor):
            logits = self.linear(self.pooler(x).squeeze(1)).unsqueeze(1)
        else:
            logits = torch.stack(
                [self.linear(self.pooler(segment).squeeze(1)) for segment in x],
                dim=1,
            )
        mask = _valid_mask(segment_mask, logits)
        return self._aggregate(logits, mask)

    def _aggregate(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mode = self.aggregation
        if mode in {"mean_logit", "logit_mean"}:
            return _masked_mean(logits, mask)

        probs = logits.float().sigmoid()
        if mode in {"learned_stats", "stats_mlp"}:
            features = _window_stats_features(
                logits=logits.float(),
                probs=probs,
                mask=mask,
                percentiles=self.stats_percentiles,
                topks=self.stats_topk,
                source=self.stats_source,
            ).to(dtype=logits.dtype)
            return self.stats_mlp(features)
        if mode in {"mean_prob", "prob_mean"}:
            agg_probs = _masked_mean(probs, mask)
        elif mode in {"max_prob", "prob_max"}:
            agg_probs = _masked_max(probs, mask)
        elif mode in {"topk_prob", "prob_topk"}:
            agg_probs = _masked_topk_mean(probs, mask, self.topk)
        else:
            raise ValueError(f"Unsupported WindowMILCORNHead aggregation={mode!r}")

        agg_probs = agg_probs.clamp(1e-4, 1.0 - 1e-4).to(dtype=logits.dtype)
        return torch.logit(agg_probs)


def _valid_mask(segment_mask: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor:
    if segment_mask is None:
        return torch.ones(logits.shape[:2], device=logits.device, dtype=torch.bool)
    mask = segment_mask.to(device=logits.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(logits.shape[:2]):
        raise ValueError(f"segment_mask shape {tuple(mask.shape)} does not match {tuple(logits.shape[:2])}")
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return masked.max(dim=1).values


def _masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, topk: int) -> torch.Tensor:
    rows = []
    k = max(1, int(topk))
    for row_values, row_mask in zip(values, mask):
        valid = row_values[row_mask]
        if valid.numel() == 0:
            valid = row_values[:1]
        kk = min(k, valid.shape[0])
        rows.append(valid.topk(kk, dim=0).values.mean(dim=0))
    return torch.stack(rows, dim=0)


def _stats_feature_dim(
    num_logits: int,
    percentiles: tuple[float, ...],
    topks: tuple[int, ...],
    source: str,
) -> int:
    per_source_stats = 4 + len(percentiles) + len(topks)
    n_sources = 2 if source in {"prob_logit", "logit_prob"} else 1
    return n_sources * int(num_logits) * per_source_stats + 3


def _window_stats_features(
    logits: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
    percentiles: tuple[float, ...],
    topks: tuple[int, ...],
    source: str,
) -> torch.Tensor:
    sources = []
    if source in {"prob", "probs", "probability", "prob_logit", "logit_prob"}:
        sources.append(probs)
    if source in {"logit", "logits", "prob_logit", "logit_prob"}:
        sources.append(logits)
    if not sources:
        raise ValueError(f"Unsupported WindowMILCORNHead stats_source={source!r}")

    rows = []
    for source_values, row_mask in zip(zip(*sources), mask):
        valid_sources = [values[row_mask] for values in source_values]
        if valid_sources[0].numel() == 0:
            valid_sources = [values[:1] for values in source_values]

        features = []
        for valid in valid_sources:
            features.extend(
                [
                    valid.mean(dim=0),
                    valid.std(dim=0, unbiased=False),
                    valid.min(dim=0).values,
                    valid.max(dim=0).values,
                ]
            )
            sorted_valid = valid.sort(dim=0).values
            n = sorted_valid.shape[0]
            for percentile in percentiles:
                idx = int(round(max(0.0, min(1.0, float(percentile))) * (n - 1)))
                features.append(sorted_valid[idx])
            for topk in topks:
                kk = min(max(1, int(topk)), n)
                features.append(valid.topk(kk, dim=0).values.mean(dim=0))

        valid_count = row_mask.sum().to(dtype=logits.dtype)
        total_count = torch.tensor(row_mask.numel(), device=logits.device, dtype=logits.dtype)
        length_features = torch.stack(
            [
                valid_count,
                torch.log1p(valid_count),
                valid_count / total_count.clamp_min(1.0),
            ]
        )
        rows.append(torch.cat([*features, length_features], dim=0))
    return torch.stack(rows, dim=0)
