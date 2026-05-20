# PD Hand V-JEPA 2.1 Training Runbook

This runbook documents the one-fold PD hand-task probe run for `item_3_4` on V-JEPA 2.1.

## Files

- Slurm submit script: `scripts/pd_hand/run_item_3_4_fold0_wandb_ddp3.sbatch`
- Spearman-targeted Slurm script: `scripts/pd_hand/run_item_3_4_fold0_spearman_wandb_ddp3.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_wandb_ddp3.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense softmax Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_wandb_ddp3.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense balanced-softmax Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_balanced_wandb_ddp3.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense balanced-softmax no-smoothing Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_balanced_nosmooth_wandb_ddp3.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense softmax no-augmentation Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_noaug_wandb_ddp2.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense softmax no-augmentation/no-crop Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_noaug_nocrop_wandb_ddp2.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense stats-head balanced no-augmentation/no-crop Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_stats_balanced_noaug_nocrop_wandb_ddp2.sbatch`
- W&B auth preflight: `scripts/pd_hand/prepare_wandb_auth.sh`
- Monitor helper: `scripts/pd_hand/monitor_item_3_4_fold0.sh`
- V-JEPA 2.1 multiclip wrapper: `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_vjepa21.py`
- Repo-local config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl_ssv2_corn.yaml`
- Spearman-targeted config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl_ssv2_corn_spearman.yaml`
- Accuracy-targeted dense config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml`
- Accuracy-targeted dense softmax config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_accuracy_dense.yaml`
- Accuracy-targeted dense balanced-softmax config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_balanced_accuracy_dense.yaml`
- Accuracy-targeted dense balanced-softmax no-smoothing config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_balanced_nosmooth_accuracy_dense.yaml`
- Accuracy-targeted dense softmax no-augmentation config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_noaug_accuracy_dense.yaml`
- Accuracy-targeted dense softmax no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_noaug_nocrop_accuracy_dense.yaml`
- Accuracy-targeted dense stats-head balanced no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_stats_balanced_noaug_nocrop_accuracy_dense.yaml`
- Current output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_ddp3_h100x3`
- Spearman-targeted output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_spearman_h100x3`
- Accuracy-targeted dense output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_h100x3`
- Accuracy-targeted dense softmax output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_h100x3`
- Accuracy-targeted dense balanced-softmax output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_balanced_h100x3`
- Accuracy-targeted dense balanced-softmax no-smoothing output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_balanced_nosmooth_h100x3`
- Accuracy-targeted dense softmax no-augmentation output root: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_noaug_h100x2`
- Current Slurm logs: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0/run_logs/`
- Current external split CSVs: `/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/splits/item_3_4/`

## W&B

The API key file is intentionally not committed:

```bash
/gpfs/milgram/pi/scherzer/yl2428/vjepa2/wandb_api.txt
```

The Slurm script calls `scripts/pd_hand/prepare_wandb_auth.sh` before `srun`. That script first verifies the existing `video-llama` W&B login. If the older W&B SDK cannot read the scoped key directly, it bootstraps `/home/yl2428/.netrc` with a temporary newer W&B install, then verifies the `video-llama` env again.

Expected Slurm stderr evidence:

```text
wandb: Currently logged in as: yl2428 to https://api.wandb.ai.
wandb: Tracking run with wandb version 0.21.0
wandb: Run `wandb offline` to turn off syncing.
wandb: Resuming run pd-hand-item-3_4-fold-0
wandb: View run at https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0
```

The unweighted offline warmup run was synced to:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/vipsjot6
```

The online resumed run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0
```

The Spearman-targeted online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-spearman
```

The accuracy-targeted dense online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-pos4096
```

The accuracy-targeted dense softmax online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax
```

The accuracy-targeted dense balanced-softmax online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced
```

The accuracy-targeted dense balanced-softmax no-smoothing online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced-nosmooth
```

The accuracy-targeted dense softmax no-augmentation online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug
```

The accuracy-targeted dense softmax no-augmentation/no-crop online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug-nocrop
```

## Submit

From the repo root:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_wandb_ddp3.sbatch
```

The script requests one node with 3 H100s and resumes from `latest.pt` if present. It uses `WANDB_MODE=online`, does not export the key as `WANDB_API_KEY`, and lets W&B use verified `.netrc` auth.

For a same-fold run that selects heads and checkpoints by Spearman instead of QWK:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_spearman_wandb_ddp3.sbatch
```

For the corrected V-JEPA 2.1 dense run targeting validation accuracy `>= 70%`:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_wandb_ddp3.sbatch
```

For the softmax comparison run using the same dense V-JEPA 2.1 sampling:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_wandb_ddp3.sbatch
```

For the balanced softmax comparison run:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_balanced_wandb_ddp3.sbatch
```

For the balanced softmax comparison without label smoothing:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_balanced_nosmooth_wandb_ddp3.sbatch
```

For the dense softmax no-augmentation comparison, currently sized for two available H100s:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_noaug_wandb_ddp2.sbatch
```

For the same no-augmentation comparison without MediaPipe hand cropping:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_softmax_noaug_nocrop_wandb_ddp2.sbatch
```

For the lower-variance stats-pooling head on the no-augmentation/no-crop input:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_stats_balanced_noaug_nocrop_wandb_ddp2.sbatch
```

## Monitor

```bash
scripts/pd_hand/monitor_item_3_4_fold0.sh <job_id>
```

For the active fixed dense run, use the accuracy-targeted command below with job id `28862001`.

The monitor prints Slurm state, recent train/validation logs, W&B stderr lines, CSV rows, `metrics_latest.json`, confusion matrix, and coverage.

For the Spearman-targeted run:

```bash
TAG=pd-hand-item-3_4-fold-0-spearman \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_spearman_h100x3/video_classification_frozen/pd-hand-item-3_4-fold-0-spearman \
JOB_NAME=vjepa_i34_f0_sp3 \
LOG_PREFIX=spearman_ddp3 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28861962
```

For the accuracy-targeted dense run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-pos4096 \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_h100x3/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-pos4096 \
JOB_NAME=vjepa_i34_f0_acc21 \
LOG_PREFIX=acc_vjepa21_dense_ddp3 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862001
```

For the accuracy-targeted dense softmax run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_h100x3/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax \
JOB_NAME=vjepa_i34_f0_sm21 \
LOG_PREFIX=acc_vjepa21_dense_softmax_ddp3 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862029
```

For the accuracy-targeted dense balanced-softmax run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_balanced_h100x3/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced \
JOB_NAME=vjepa_i34_f0_smb21 \
LOG_PREFIX=acc_vjepa21_dense_softmax_balanced_ddp3 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862032
```

For the accuracy-targeted dense balanced-softmax no-smoothing run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced-nosmooth \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_balanced_nosmooth_h100x3/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-balanced-nosmooth \
JOB_NAME=vjepa_i34_f0_smbn21 \
LOG_PREFIX=acc_vjepa21_dense_softmax_balanced_nosmooth_ddp3 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862033
```

For the accuracy-targeted dense softmax no-augmentation run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_noaug_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug \
JOB_NAME=vjepa_i34_f0_smna21 \
LOG_PREFIX=acc_vjepa21_dense_softmax_noaug_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862035
```

For the accuracy-targeted dense softmax no-augmentation/no-crop run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug-nocrop \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_softmax_noaug_nocrop_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug-nocrop \
JOB_NAME=vjepa_i34_f0_smnc21 \
LOG_PREFIX=acc_vjepa21_dense_softmax_noaug_nocrop_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862038
```

## Current Run State

Historical note from May 19, 2026 21:19 EDT: job `28861909` was the active weighted CORN online run on `r818u35n11`. It resumed from epoch 3 and started epoch 4 with:

```text
corn_pos_weight: [0.4502924, 1.9176470, 12.0526314, 1.0]
```

These weights come from fold 0 train labels:

```text
{0: 77, 1: 86, 2: 66, 3: 19}
```

Validation has:

```text
{0: 19, 1: 20, 2: 22, 3: 4, 4: 1}
```

The first three epochs were unweighted and did not show ordinal separation. Epoch 4 was weighted but still had QWK `0.0`. Epochs 5-8 show a weak but real positive ordinal signal after weighting:

```text
epoch 5: val_spearman 0.23088, val_qwk 0.08375, val_mae 0.77273
epoch 6: val_spearman 0.19113, val_qwk 0.07373, val_mae 0.80303
epoch 7: val_spearman 0.12704, val_qwk 0.13343, val_mae 0.81818
epoch 8: val_spearman 0.11729, val_qwk 0.19581, val_mae 0.74242
```

Epoch 8 confusion matrix:

```text
[[0, 18, 1, 0, 0],
 [0, 20, 0, 0, 0],
 [0, 21, 1, 0, 0],
 [0,  3, 1, 0, 0],
 [0,  0, 0, 1, 0]]
```

This was useful historical evidence that the weighted CORN objective could learn some score separation: QWK was positive for four consecutive weighted epochs, QWK improved to `0.19581` by epoch 8, Spearman stayed positive, MAE improved, and predictions were no longer a single column.

As of May 20, 2026 03:50 EDT, the active target is validation accuracy `>= 70%`.

- QWK-selected job `28861909` and Spearman-targeted job `28861962` were stopped because they used the older wrapper and were well below the target.
- Accuracy-targeted dense job `28861995` reached epoch 1 batch 60, then failed because one clip needed temporal index `603` while `max_frames: 1024` only created 512 temporal-token positions.
- Fixed accuracy-targeted dense CORN job `28862001` reached epoch 2 validation accuracy `34.84849` and was stopped to free GPUs for softmax comparisons.
- Dense softmax comparison job `28862029` reached epoch 1 validation accuracy `30.30303` and was stopped.
- Dense balanced-softmax comparison job `28862030` failed before the first batch due a weighted-loss autocast dtype mismatch; the loss was fixed and resubmitted as job `28862032`. Job `28862032` reached epoch 1 validation accuracy `30.30303`, stayed weak in epoch 2 training, and was stopped to free the GPU/QOS budget for the no-augmentation run.
- Dense balanced-softmax no-smoothing comparison job `28862033` is running. It keeps `class_weight: balanced` but removes label smoothing to test direct raw-accuracy optimization. Fold-0 train has no class-4 examples, so the auto class-4 training weight is `0.0`. Epoch 1 reached validation accuracy `36.36364`, Spearman `0.31613`, QWK `0.14544`, MAE `0.86364`, train coverage `0.93951/0.97561`, and val coverage `0.94125/0.98255`, so it is continuing into epoch 2.
- Dense softmax no-augmentation comparison job `28862034` was stopped after the bf16 fix because it had started under the older fp16 autocast path. Replacement job `28862035` is running with 2 H100s because only two H100s were available under the current QOS limit. It keeps the dense 32-frame, stride-1, 12-segment coverage settings but disables training RandAugment and random erasing, and constrains random resized crop to `0.9-1.0` square crops.
- Future runs after the bf16 fix use true `torch.bfloat16` autocast in the video eval path. Earlier jobs used the existing video-eval behavior, where `use_bfloat16: true` actually selected fp16 autocast.
- No-MediaPipe-crop jobs `28862036` and `28862037` were stopped because they inherited the base no-augmentation W&B/config values through the shared runner. Dedicated no-crop job `28862038` is running on 2 H100s and confirmed `dataset_kwargs: {hand_crop: false}` plus W&B run id `pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug-nocrop` in its startup log.
- A stats-pooling probe head is implemented as `head_type: stats`. It pools frozen V-JEPA token features with mean/std/max and uses a small MLP, giving a lower-variance alternative to the single-query attentive probe for this small fold.

## Learning Criteria

Do not treat accuracy alone as evidence of learning. This fold is imbalanced enough that a single-class predictor can reach about 30%.

Use these as the practical checks:

- QWK becomes consistently positive and improves across epochs.
- Spearman becomes positive and remains positive.
- The confusion matrix is no longer a single predicted column.
- Predictions cover at least two adjacent score bins.
- MAE improves without collapsing predictions to one class.

## Coverage

Current temporal settings:

```yaml
frames_per_clip: 16
frame_step: 2
num_segments: 12
num_views_per_segment: 3
```

Observed epoch coverage is about:

```text
train raw-frame coverage: 54.7-54.8%
train temporal span:      97.5-97.6%
val raw-frame coverage:   53.3%
val temporal span:        97.7-97.8%
```

Raw coverage is the fraction of individual source frames sampled. Temporal span is the fraction from the earliest sampled frame to the latest sampled frame, so high span means the run touches almost the full video duration even though it does not use every raw frame.

Dense V-JEPA 2.1 accuracy run temporal settings:

```yaml
frames_per_clip: 32
frame_step: 1
num_segments: 12
num_views_per_segment: 1
wrapper_kwargs:
  max_frames: 4096
  use_pos_embed: true
```

`max_frames: 4096` gives 2048 temporal-token positions with tubelet size 2, enough for clips well beyond the nominal 20 second / 30 fps case. The wrapper also regenerates a fixed sine/cosine table on the fly if a sampled clip exceeds the configured range.

Preflight on May 20, 2026 confirmed strict V-JEPA 2.1 checkpoint loading with all keys matched. A deterministic validation sampling smoke test on the first three fold-0 validation clips produced raw-frame coverage between `96.7%` and `99.7%`, with matching repeated indices for the same sample.

## If The Weighted Run Still Collapses

If later epochs collapse back to QWK `0.0` with a single-column confusion matrix, the next clean move is to restart from epoch 0 with `corn_pos_weight: auto` instead of resuming heads that were already biased by the unweighted warmup.
