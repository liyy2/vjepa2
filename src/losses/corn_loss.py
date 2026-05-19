"""CORN ordinal loss for 0..K-1 labels.

CORN represents a K-level ordinal target with K-1 binary logits. Logit j
answers whether the target is greater than j.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_targets(labels: torch.Tensor, num_levels: int) -> torch.Tensor:
    """Convert integer ordinal labels to CORN binary targets."""
    labels = labels.long()
    thresholds = torch.arange(num_levels - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_levels: int = 5,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy over the K-1 CORN threshold logits."""
    if logits.ndim != 2 or logits.shape[1] != num_levels - 1:
        raise ValueError(
            f"Expected logits with shape [B, {num_levels - 1}], got {tuple(logits.shape)}."
        )
    labels = labels.long()
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_levels):
        raise ValueError(f"Labels must be in [0, {num_levels - 1}].")
    targets = corn_targets(labels, num_levels=num_levels)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def corn_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return P(y=k) for each ordinal level implied by CORN logits."""
    conditional_probs = torch.sigmoid(logits)
    batch_size, num_thresholds = conditional_probs.shape
    num_levels = num_thresholds + 1
    greater_than_probs = torch.cumprod(conditional_probs, dim=1)
    probs = logits.new_empty((batch_size, num_levels))
    probs[:, 0] = 1.0 - greater_than_probs[:, 0]
    for level in range(1, num_thresholds):
        probs[:, level] = greater_than_probs[:, level - 1] - greater_than_probs[:, level]
    probs[:, -1] = greater_than_probs[:, -1]
    return probs.clamp_min(0.0)


def corn_expected_score(logits: torch.Tensor) -> torch.Tensor:
    """Expected ordinal score from CORN logits."""
    return torch.cumprod(torch.sigmoid(logits), dim=1).sum(dim=1)


def corn_predict_label(logits: torch.Tensor) -> torch.Tensor:
    """Integer prediction from CORN logits."""
    return (torch.sigmoid(logits) > 0.5).long().sum(dim=1)
