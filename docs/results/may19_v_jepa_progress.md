# may19 finger-tap V-JEPA progress — item 3.4 fold 0

Date: 2026-05-22

## Dataset

| Stat | may19 | old |
|---|---:|---:|
| Total clips (ok, labeled) | 1023 | 640 |
| Item 3.4 clips | 511 | 320 |
| Subjects (item 3.4) | 244 | 154 |
| Fold 0 train clips | 404 | 248 |
| Fold 0 val clips | 107 | 66 |
| Class distribution (item 3.4) | {0:180, 1:166, 2:127, 3:37, 4:1} | {0:96, 1:106, 2:88, 3:23, 4:1} |

Manifest: `outputs/foundation_model_minimal_hand_tasks_may19/hand_clip_manifest.csv`
Splits: `outputs/foundation_model_minimal_hand_tasks_may19/splits/item_3_4/fold_*_{train,val}.csv`

## Reference numbers, may19 fold 0

| Approach | Acc | QWK | MAE | Notes |
|---|---:|---:|---:|---|
| Kinematic LightGBM (ExtraTrees old_118) | 71.96 | **0.728** | 0.355 | Target |
| Frozen V-JEPA combiner — fine-only, d=192 (best of sweep) | 50.47 | 0.476 | 0.579 | 2.0M params, early-stopped epoch 2 |
| Frozen V-JEPA combiner — fine+coarse aug, d=256 | 41.12 | 0.464 | 0.757 | mixup 0.2 + seg-dropout 0.1 |
| Frozen V-JEPA combiner — fine-only, d=256 (basic) | 41.12 | 0.464 | 0.757 | first run; full 250-epoch budget |
| Frozen V-JEPA TemporalStatsMLP — best single (192 cfgs) | 47.66 | 0.515 | 0.729 | partial sweep best; seed 98, balanced CE, no FFT, velocity on |
| Frozen V-JEPA TemporalStatsMLP — best top-K=24 ensemble | 48.60 | **0.533** | 0.682 | full sweep top-K probability average |
| Frozen V-JEPA combiner — tiny d=128 L=2 seed 0 (best of tinysweep) | 52.34 | 0.510 | 0.570 | mixup 0.2 + seg-dropout 0.2 + token-dropout 0.1, 0.54M params, early-stop epoch 6 |
| Frozen V-JEPA LightGBM (fine PCA64 + summary stats) | 45.79 | 0.322 | 0.710 | Multiclass; regression variant worse |
| Frozen V-JEPA LightGBM (fine + coarse multi-scale regression) | 30.84 | 0.149 | 0.776 | Worse than fine-only; tree models limit |
| Supervised LoRA-FT V-JEPA — corn auto pos_weight (FAILED) | 27.10 | **0.000** | 0.804 | Collapsed; class-prior pred. Root cause: level-3 pos_weight = 403 (only 1 sample at class 4) |

The frozen V-JEPA features hit a ~0.48 QWK ceiling on may19 fold 0 regardless of head (combiner, stats MLP). Gap to kinematic baseline: ~0.25 QWK. Confirms the diagnosis in `vjepa_appearance_vs_rhythm.html`: the encoder doesn't natively expose finger-tap rhythm, so any head built on frozen features hits the same ceiling.

## Architectural changes implemented

### 1. Spatial-pool / temporal-keep features (existing infra, validated)
- `scripts/pd_hand/cache_vjepa21_temporal_embeddings.py` already pools the 576 spatial patches per tubelet to one D-dim vector, KEEPING the 192 temporal token sequence per clip (12 segments × 16 tubelets) for the fine cache and 128 tokens (8 segments × 16 tubelets) for the coarse cache.
- This is what the user asked for ("squash spatial but keep temporal"). Output shape per clip is `[T, D]`, T = 128 or 192 depending on cache.

### 2. Cross-window combiner (replaces logit-mean across the multi-clip eval grid)
- New: `scripts/pd_hand/train_corn_combiner.py`
- Architecture: small transformer (d_model 128–384, 2–4 layers, 4–8 heads) over the full per-clip temporal token sequence.
- 2D learned positional encoding: separate embeddings for (segment_idx, within_segment_idx) so the model knows BOTH which 1-second window a token came from AND its position within the window.
- Learnable CLS token reads the clip representation.
- CORN ordinal head (binary logits over K−1 = 4 levels) with per-level pos_weight derived from conditional-set counts, clipped to [1/4, 4] to avoid the level-3 explosion that killed supervised LoRA-FT.
- Supports multi-scale input via per-scale projection + per-scale stream embedding + concat-along-time.

### 3. Multi-scale fine/coarse tubelet caches (replaces "logits averaged across 12 segments")
- Fine cache: `frame_step=1, frames_per_clip=32, num_segments=12` → 192 tubelet tokens per clip @ 33ms/token
- Coarse cache: `frame_step=2, frames_per_clip=32, num_segments=8` → 128 tubelet tokens per clip @ 67ms/token, 2.14s temporal coverage per window
- Combiner takes both, fuses them with stream embeddings and per-scale 2D pos encoding.

### 4. Sweeps
- Frozen V-JEPA sweep (21 configs) is **done**: best QWK 0.476 (fine-only, d=192, seed 0).
- Multi-scale frozen V-JEPA sweep (12 configs) is **done**: best QWK 0.464 (fine+coarse aug, d=256, seed 1).
- TemporalStatsMLP velocity grid on may19 frozen cache (48 configs cap) — **running**.

## Per-class confusion (best tiny combiner vs kinematic, may19 fold 0)

| Class | n | Kinematic correct | V-JEPA tiny combiner correct |
|---|---:|---:|---:|
| 0 | 42 | 36 (86%) | 25 (60%) |
| 1 | 29 | 19 (66%) | 17 (59%) |
| 2 | 28 | 18 (64%) | 13 (46%) |
| 3 | 8 | 4 (50%) | **1 (12%)** |
| 4 | 0 | — | — |

V-JEPA's biggest deficits are at class 0 and class 3, the extremes of the severity spectrum. Class 3 is the canonical "decrement" class — V-JEPA almost never identifies it correctly, which directly confirms the appearance-vs-rhythm diagnosis.

## Queued / in flight

| Job | Stage | Status |
|---:|---|---|
| 28870252 | TemporalStatsMLP velocity grid on may19 frozen cache | RUNNING |
| 28870263 | Supervised LoRA-FT on may19 fold 0 (rerun with `corn_pos_weight: null`) | PENDING |
| 28870264 | Merge LoRA-FT checkpoint | PENDING (afterok 28870263) |
| 28870265 | Cache fine from LoRA-FT backbone | PENDING (afterok 28870264) |
| 28870266 | Cache coarse from LoRA-FT backbone | PENDING (afterok 28870264) |
| 28870267 | Combiner sweep on LoRA-FT fine cache | PENDING (afterok 28870265) |
| 28870268 | Combiner sweep on LoRA-FT multi-scale cache | PENDING (afterok 28870265+28870266) |

## Lessons learned

1. **`corn_pos_weight: auto` is unsafe** on this dataset because class 4 has 1 sample → level-3 pos_weight ≈ 403 → forces over-positive prediction at level 3 → "always predict class 1" collapse. Replacement (`null` or capped per-level weighting in my combiner code) fixes this.
2. **Best frozen-V-JEPA combiner is small (d=192) and early-stopped at epoch 2.** With 404 train clips, the larger models overfit immediately. This suggests the bottleneck is feature quality, not head capacity.
3. **Multi-scale didn't materially help** over fine-only on frozen V-JEPA. Likely because both scales come from the same frozen encoder with the same appearance bias — they don't add motion information that wasn't already encoded somewhere in the fine cache. LoRA-FT may change this.
4. **Stats MLP with FFT + velocity ≈ Combiner.** Suggests rhythm info IS in the embeddings but at a level the head can recover similarly with either approach.

## Next moves if LoRA-FT chain works

1. Compare LoRA-FT cache combiner vs frozen cache combiner; expect +0.05-0.15 QWK
2. Multi-scale on LoRA-FT cache: if this works, expect +0.03-0.05 over single-scale LoRA-FT
3. If still < 0.6 QWK, add motion-biased inputs (frame difference cache)
4. Once we exceed kinematic 0.728, replicate on folds 1–4 for honest CV

## Next moves if LoRA-FT chain fails / saturates

1. Frame-difference input cache (compute Δ_frame before V-JEPA, encode motion field)
2. Kinematic-distillation auxiliary loss during LoRA-FT
3. Subject-OOF combiner hyperparam selection inside fold-0 train
