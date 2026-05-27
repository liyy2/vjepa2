# Experiment log

This log keeps the useful result history and removes stale exploratory reports.

## Timeline

| Date | Finding | Status |
| --- | --- | --- |
| 2026-05-20 | Early item 3.4 V-JEPA probes were below the inflated kinematic baseline. | Superseded by leakage audit. |
| 2026-05-21 | Code scan found runtime and protocol risks: encoder naming mismatch, stale checkpoint guards, metadata handling, short-video indexing, and fragile distributed env setup. | Useful engineering context only. |
| 2026-05-22 | Adaptive-window and LoRA recovery work stabilized cache generation and preserved progress after queue/OOM issues. | LoRA utilities retained; old run sprawl removed. |
| 2026-05-23 | R2 and representation scans did not show V-JEPA recovering detailed kinematic features. A validation-tuning path was identified as unsafe for clean claims. | Removed from headline path. |
| 2026-05-24 | Leakage audit showed the old strong baseline used diagnosis and item 3.5-like information. Re-running against fair `kin_only` gave `0.3615 -> 0.4090`, delta `+0.0475`, 4/5 folds positive, 35/50 seed-fold wins. | Current positive result. |
| 2026-05-25 | Audit caveats: subject grouping and OOF protocol were mostly right, but feature inclusion needed explicit code guards, `clip_path` joins needed to fail loudly, and "50 observations" should be described as fold x seed pairs, not subjects. | Addressed in cleaned scripts and docs. |
| 2026-05-27 | Cleanup pass removed exploratory scripts/configs/docs, kept the minimal May 24 reproduction chain, and retained LoRA/CORN/temporal-head utilities. | Current repository state. |

## Current interpretation

The May 24 result supports a narrow claim:

> On item 3.4, V-JEPA OOF probabilities add complementary signal to fair
> kinematic-only video features in this 5-fold subject-disjoint evaluation.

Do not claim:

- V-JEPA beats a diagnosis-aware clinical baseline.
- V-JEPA-only features beat kinematics.
- The result is fold-level significant; fold-mean Wilcoxon over 5 folds is `p=0.09375`.
- The 50 paired observations are independent subjects; they are 5 folds x 10 seeds.

## Retained code surface

May 24 reproduction:

- `scripts/pd_hand/cache_vjepa21_temporal_embeddings.py`
- `scripts/pd_hand/oof_vjepa_combiner_predictions.py`
- `scripts/pd_hand/augment_with_oof_probs.py`
- `scripts/pd_hand/hybrid_multiseed.py`
- `scripts/pd_hand/aggregate_kinonly_5fold.py`
- `scripts/pd_hand/run_may24_cache_vjepa_features.sbatch`
- `scripts/pd_hand/run_may24_oof_vjepa_probs.sbatch`
- `scripts/pd_hand/run_may24_augment_oof_features.sbatch`
- `scripts/pd_hand/run_may24_fair_multiseed.sbatch`

Useful non-headline infrastructure:

- LoRA: `launch_lora_jepa.py`, `merge_lora_checkpoint.py`, `merge_lora_from_base.py`, retained LoRA sbatch/config templates
- CORN: `train_corn_combiner.py`
- temporal head: `train_vjepa21_temporal_encoder.py`
- kinematic feature generation: `train_finger_tapping_kinematic_baseline.py`
