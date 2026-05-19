"""Compatibility wrapper for CORN ordinal losses."""

from src.losses.corn_loss import corn_expected_score, corn_loss, corn_predict_label, corn_probabilities, corn_targets

__all__ = [
    "corn_expected_score",
    "corn_loss",
    "corn_predict_label",
    "corn_probabilities",
    "corn_targets",
]

