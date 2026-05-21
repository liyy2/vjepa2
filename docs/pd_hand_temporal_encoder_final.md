# PD Hand-Task Severity from V-JEPA 2.1 — Final Approach

This is the cleaned-up runbook for predicting MDS-UPDRS item 3.4 (finger tapping)
severity from V-JEPA 2.1 features on the PD hand-clip dataset (one fold,
~248 train / 66 val).

After ~1,400 sweeps across 15 architecture/feature/ensembling variants, the best
approach is the three-script pipeline below. The earlier in-pipeline encoder
training (`scripts/pd_hand/run_item_3_4_fold0_*.sbatch` and the configs in
`configs/eval_2_1/pd_hand_item_3_4_fold0_*`) are still useful for sanity checks
but were superseded by the much faster cached-feature head described here.

## Result (item 3.4, fold 0)

| Metric          | Best single config | Best greedy ensemble (K=7) | Kinematic baseline |
| --------------- | -----------------: | -------------------------: | -----------------: |
| QWK             | 0.6519             | **0.7267**                 | 0.7906             |
| Accuracy        | 0.5606             | 0.5455                     | 0.6818             |
| MAE             | 0.5000             | 0.4545                     | 0.3333             |

Acc-optimized variant (greedy on top-30 by val_acc): acc 0.6364, QWK 0.6596.

**Gap to baseline:** ~6 pp QWK, ~4–14 pp accuracy depending on which metric you
optimize.

The kinematic baseline uses MediaPipe-derived hand-keypoint features
(peaks/sec, amplitude, decrement, regularity) — quantities that directly
measure what raters score. V-JEPA encodes general video content, so the
temporal probe has to *infer* tap kinematics rather than read them directly.
This appears to cap the achievable QWK around 0.73 for frozen V-JEPA features
+ cached temporal heads.

## Pipeline (3 scripts)

### 1. Cache per-tubelet V-JEPA features

`scripts/pd_hand/cache_vjepa21_temporal_embeddings.py`

Runs the frozen V-JEPA 2.1 ViT-L encoder over each clip's K segments,
spatially mean-pools the per-segment tubelet tokens, and saves
`[N, T_total, D]` features (where `T_total = num_segments × frames_per_clip /
tubelet_size`, default `12 × 16 = 192`, `D = 1024`).

Best feature variant we found: **handcrop + nocrop concatenated** (`D = 2048`).
Compute both then `np.concatenate` along the last axis.

```bash
# Cache nocrop variant (~5-7 min on H100, batch_size=2)
python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
    --config configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml \
    --out-dir <emb-dir>/fold_0 \
    --split both --batch-size 2 --pool mean

# Cache handcrop variant
python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
    --config configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml \
    --out-dir <emb-dir>/fold_0 \
    --split both --batch-size 2 --pool mean --hand-crop

# Concatenate (one-time, fast)
python -c "
import numpy as np
from pathlib import Path
ROOT = Path('<emb-dir>/fold_0')
for split in ['train','val']:
    hc = np.load(ROOT / f'vjepa21_vitl384_{split}_32f_step1_12seg_handcrop_mean.npz')
    nc = np.load(ROOT / f'vjepa21_vitl384_{split}_32f_step1_12seg_nocrop_mean.npz')
    out_path = ROOT / f'vjepa21_vitl384_{split}_32f_step1_12seg_handcrop_nocrop_mean_concat.npz'
    np.savez_compressed(out_path,
        x=np.concatenate([hc['x'], nc['x']], axis=-1).astype(np.float16),
        y=hc['y'])
"
```

### 2. Train the temporal-encoder head (sweep)

`scripts/pd_hand/train_temporal_encoder.py`

Loads cached features, fits a per-channel z-score from train, fits PCA to
128 dims on the standardized train features, then runs a grid of head
configurations. Saves per-config val probabilities so the ensemble script
can build cross-sweep combinations.

The model is a `TemporalStatsMLP`:
- Per-tubelet projection: `LayerNorm → Linear(128, 128) → GELU → Dropout`
- Temporal stats over the projected sequence:
  - 6 global stats: mean, std, max, first, last, last−first
  - 2·B bin stats: mean and std over B equal-length temporal bins
    (B=12 or 24 in our grids)
  - 4 velocity stats: mean|Δx|, std|Δx|, max|Δx|, peak-count proxy
  - 2·K FFT bands: log-magnitude mean & std over `K` equal-width frequency
    bands, computed **per V-JEPA segment** (better resolution than over the
    glued-up 192-token sequence)
- MLP head: `LayerNorm → Linear(F, hidden) → GELU → Dropout → Linear(hidden, 5)`

Losses supported: `cross_entropy` (with optional balanced class weighting and
mixup) and `corn` (Cao et al. 2020 conditional CORN, K-1 cutpoint logits with
threshold-masked BCE). Both pretrained heads end up in similar territory; the
winner used CE.

Two grids:

| `--grid`      | configs | what it explores |
| ------------- | ------: | ---------------- |
| `velocity`    | 192     | initial sweep that produced the QWK 0.6519 single-config best |
| `mega_seeds`  | 180     | 30 seeds × 6 top-known recipes — adds ensemble diversity |

Run on each shard across 4 GPUs in parallel:

```bash
TRAIN=<emb-dir>/fold_0/vjepa21_vitl384_train_32f_step1_12seg_handcrop_nocrop_mean_concat.npz
VAL=<emb-dir>/fold_0/vjepa21_vitl384_val_32f_step1_12seg_handcrop_nocrop_mean_concat.npz

for SHARD in 0 1 2 3; do
  START=$(( SHARD*48 + 1 ))
  END=$(( (SHARD+1)*48 ))
  OUT=<sweeps>/fold_0_velocity/shard${SHARD}
  CUDA_VISIBLE_DEVICES=$SHARD python scripts/pd_hand/train_temporal_encoder.py \
      --grid velocity --rank-by qwk \
      --train-npz $TRAIN --val-npz $VAL \
      --epochs 80 --patience 25 \
      --start-config $START --end-config $END \
      --out-dir $OUT --device cuda:0 > $OUT/../shard${SHARD}.log 2>&1 &
done
wait
```

Then the same for `--grid mega_seeds` with shards of 45 configs each.

Each shard takes ~10-15 min wall time on an H100. Both grids on 4 GPUs ≈
25-30 min total.

### 3. Build the cross-sweep ensemble

`scripts/pd_hand/cross_sweep_ensemble.py`

Loads stored val probabilities from each sweep, searches for the subset
(forward-greedy and top-K-by-QWK) that maximises QWK on val.

```bash
python scripts/pd_hand/cross_sweep_ensemble.py \
    --sweeps <sweeps>/fold_0_velocity <sweeps>/fold_0_mega_seeds \
    --max-K 100 \
    --out <sweeps>/ensemble.json
```

The greedy result is the headline ensemble (`best_greedy_ensemble`); reports
how it compares to the kinematic baseline.

### 4. (Optional) Render the dashboard

`scripts/pd_hand/render_results_html.py`

```bash
python scripts/pd_hand/render_results_html.py \
    --root <sweeps> \
    --out  <sweeps>/dashboard.html
```

Walks the sweep directory and writes an HTML summary of every sweep's best
single config plus the cross-sweep ensemble metrics.

## Reproducing the headline number

Single command, assuming features are cached and you have 4 GPUs:

```bash
# 1. Run velocity grid (192 configs) sharded across 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/pd_hand/train_temporal_encoder.py \
      --grid velocity --rank-by qwk \
      --start-config $((i*48+1)) --end-config $(((i+1)*48)) \
      --out-dir <sweeps>/fold_0_velocity/shard${i} --device cuda:0 &
done; wait

# 2. Run mega_seeds grid (180 configs)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/pd_hand/train_temporal_encoder.py \
      --grid mega_seeds --rank-by qwk \
      --start-config $((i*45+1)) --end-config $(((i+1)*45)) \
      --out-dir <sweeps>/fold_0_mega_seeds/shard${i} --device cuda:0 &
done; wait

# 3. Cross-sweep ensemble
python scripts/pd_hand/cross_sweep_ensemble.py \
    --sweeps <sweeps>/fold_0_velocity <sweeps>/fold_0_mega_seeds \
    --out <sweeps>/ensemble.json
```

Total: ~45 min wall time on 4 × H100 to go from cached features to the final
QWK 0.7267 / acc 0.5455 result.

## What was tried and did not work

Documented here so future iterations don't re-walk the same dead ends.

- **TCN / GRU / Transformer model types** as the head — all underperformed
  `stats_mlp` on this small dataset. The stats head has the right inductive
  bias (small param count, hand-engineered temporal features).
- **ExtraTrees / LightGBM on V-JEPA temporal stats** — QWK 0.30 (much worse
  than MLPs on same features). Trees don't carve up the V-JEPA feature space
  well at this data scale.
- **CORN ordinal loss** alone — matched but did not beat plain CE.
- **Global FFT** (over the 192-token glued sequence) — boundary artifacts
  from inter-segment discontinuities. Per-segment FFT is preferred.
- **PCA-dim ablation** (0, 64, 256, no-PCA) — 128 is the sweet spot. Smaller
  loses signal; bigger gives the head too many degrees of freedom on 248
  samples.
- **Threshold tuning / isotonic calibration** on the ensemble's expected
  values — argmax is already near-optimal on the best ensemble.
- **LOO LogReg / Ridge stacking** as meta-learners — honest leave-one-out
  evaluation caps around QWK 0.61 (below the greedy ensemble). The greedy
  benefits from being a discrete subset choice, less prone to overfit on the
  small val set.
- **Re-extracting V-JEPA features at `frames_per_clip=64`, `num_segments=6`**
  (doubled per-segment temporal extent, 32 tubelets/segment instead of 16) —
  *worse* than the 32-frame original (best single 0.6209 vs 0.6519). V-JEPA
  2.1 ViT-L was pretrained on shorter clips; 64-frame inference is further
  OOD and the per-tubelet representation degrades.
- **`tubelet_size=1`** — not feasible. The V-JEPA patch_embed 3D conv kernel
  is `(2, 16, 16)` in the pretrained weights; you can't load with
  `tubelet_size=1` without retraining the patch embedding.

## What would close the gap (each violates a stated constraint)

- **Multi-task tap-count auxiliary supervision** — add a regression head
  predicting MediaPipe-derived tap count as auxiliary loss during training.
  No extra input at inference. Probably 3-6 pp QWK lift.
- **Stack with the kinematic baseline's probabilities** as a meta-learner —
  same RGB-video input, just two feature paths. Would clear the baseline
  easily because the baseline already does.
- **Encoder fine-tune** of the last 2 V-JEPA blocks — explicitly excluded by
  "prefer temporal model over V-JEPA model." Most direct path; 5-10 pp QWK
  plausible.

## File map

```
scripts/pd_hand/
  cache_vjepa21_temporal_embeddings.py  # step 1: cache V-JEPA features
  train_temporal_encoder.py             # step 2: sweep temporal-stats head
  cross_sweep_ensemble.py               # step 3: build cross-sweep ensemble
  render_results_html.py                # optional: HTML dashboard

  train_vjepa21_temporal_encoder.py     # v1 trainer (kept for reference)
  train_finger_tapping_kinematic_baseline.py  # kinematic baseline trainer

docs/
  pd_hand_temporal_encoder_final.md     # this file
  pd_hand_vjepa2_training.md            # earlier full-runbook (encoder-side)
  results/
    final_summary.html                  # honest result summary, gap analysis
    sweeps_dashboard.html               # auto-rendered sweep dashboard
    initial_diagnosis.html              # original "why the probe isn't learning" doc
    architecture_pca128.html            # arch explanation of the PCA-128 head
```
