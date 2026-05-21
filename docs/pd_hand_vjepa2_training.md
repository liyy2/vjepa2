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
- Accuracy-targeted V-JEPA 2.1 dense stats-head metadata balanced no-augmentation/no-crop Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_balanced_noaug_nocrop_wandb_ddp2.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense stats-head metadata expected-round no-augmentation/no-crop Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_eround_balanced_noaug_nocrop_wandb_ddp2.sbatch`
- Accuracy-targeted V-JEPA 2.1 dense stats-head scaled-metadata no-augmentation/no-crop Slurm script: `scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_scaled_noaug_nocrop_wandb_ddp2.sbatch`
- Finger-tapping kinematic baseline: `scripts/pd_hand/train_finger_tapping_kinematic_baseline.py`
- V-JEPA 2.1 temporal embedding cache: `scripts/pd_hand/cache_vjepa21_temporal_embeddings.py`
- V-JEPA 2.1 cached temporal encoder sweep: `scripts/pd_hand/train_vjepa21_temporal_encoder.py`
- W&B auth preflight: `scripts/pd_hand/prepare_wandb_auth.sh`
- Monitor helper: `scripts/pd_hand/monitor_item_3_4_fold0.sh`
- V-JEPA 2.1 multiclip wrapper: `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_vjepa21.py`
- Trainable V-JEPA 2.1 encoder support: `evals/video_classification_frozen/models.py`, `evals/video_classification_frozen/eval.py`
- Temporal stats pure-vision head: `src/heads/stats_head.py`
- Repo-local config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl_ssv2_corn.yaml`
- Spearman-targeted config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl_ssv2_corn_spearman.yaml`
- Accuracy-targeted dense config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml`
- Accuracy-targeted dense softmax config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_accuracy_dense.yaml`
- Accuracy-targeted dense balanced-softmax config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_balanced_accuracy_dense.yaml`
- Accuracy-targeted dense balanced-softmax no-smoothing config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_balanced_nosmooth_accuracy_dense.yaml`
- Accuracy-targeted dense softmax no-augmentation config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_noaug_accuracy_dense.yaml`
- Accuracy-targeted dense softmax no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_softmax_noaug_nocrop_accuracy_dense.yaml`
- Accuracy-targeted dense stats-head balanced no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_stats_balanced_noaug_nocrop_accuracy_dense.yaml`
- Accuracy-targeted dense stats-head metadata balanced no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_statsmeta_balanced_noaug_nocrop_accuracy_dense.yaml`
- Accuracy-targeted dense stats-head metadata expected-round no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_statsmeta_eround_balanced_noaug_nocrop_accuracy_dense.yaml`
- Accuracy-targeted dense stats-head scaled-metadata no-augmentation/no-MediaPipe-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_statsmeta_scaled_noaug_nocrop_accuracy_dense.yaml`
- Trainable last-block stats-head no-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_stats_nocrop_lastblock_ft_12seg_b2.yaml`
- Trainable last-block temporal-stats no-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_temporalstats_nocrop_lastblock_ft_12seg_b2.yaml`
- Trainable last-block temporal-stats balanced no-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_temporalstats_balanced_nocrop_lastblock_ft_12seg_b2.yaml`
- Trainable last-4-block temporal-stats no-crop config: `configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_temporalstats_nocrop_last4_ft_12seg_b1.yaml`
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

The accuracy-targeted dense stats-head metadata no-augmentation/no-crop online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-balanced-noaug-nocrop
```

The accuracy-targeted dense stats-head metadata expected-round online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-eround-balanced-noaug-nocrop
```

The accuracy-targeted dense scaled-metadata stats-head online run id is:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-scaled-noaug-nocrop
```

Cached temporal-encoder W&B runs:

```text
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/ilrd4czf   # hand-crop mean, balanced half, best 34/66
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/8kd02cml   # hand-crop mean, unweighted half, best 35/66
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/l2sy70c0   # hand-crop mean+std, unweighted half, best 33/66
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/oyuagbwf   # no-crop mean, best 37/66
https://wandb.ai/yl2428/pd-hand-vjepa2/runs/04ofw755   # hand-crop+no-crop concat, best 36/66
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

For the stats-pooling head with manifest metadata (`side` and `dx`) on the no-augmentation/no-crop input:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_balanced_noaug_nocrop_wandb_ddp2.sbatch
```

For the same metadata stats head but using rounded expected ordinal score for raw-accuracy prediction:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_eround_balanced_noaug_nocrop_wandb_ddp2.sbatch
```

For the scaled-metadata stats head:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_accuracy_vjepa21_dense_statsmeta_scaled_noaug_nocrop_wandb_ddp2.sbatch
```

For the CPU MediaPipe/kinematic baseline:

```bash
/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda run --no-capture-output -n video-llama \
  python scripts/pd_hand/train_finger_tapping_kinematic_baseline.py \
    --frame-stride 2 \
    --num-workers 8
```

Use `--force` to recompute the cached MediaPipe features.

## Cached V-JEPA Temporal Encoder

These scripts cache frozen V-JEPA 2.1 tubelet embeddings, then train small pure-vision temporal heads on the cached sequences. They do not use MediaPipe kinematic signals, `side`, `dx`, or other manifest metadata; `--hand-crop` only changes the visual crop before V-JEPA.

Cache hand-crop mean embeddings:

```bash
/home/yl2428/.conda/envs/video-llama/bin/python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
  --split both \
  --hand-crop \
  --pool mean \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda:0
```

Cache no-crop mean embeddings:

```bash
/home/yl2428/.conda/envs/video-llama/bin/python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
  --split both \
  --pool mean \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda:0
```

Train a temporal-head sweep with PCA-reduced cached embeddings and W&B logging:

```bash
WANDB_DIR=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/fold_0_nocrop \
/home/yl2428/.conda/envs/video-llama/bin/python scripts/pd_hand/train_vjepa21_temporal_encoder.py \
  --train-npz /gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0/vjepa21_vitl384_train_32f_step1_12seg_nocrop_mean.npz \
  --val-npz /gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0/vjepa21_vitl384_val_32f_step1_12seg_nocrop_mean.npz \
  --out-dir /gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/fold_0_nocrop \
  --epochs 200 \
  --batch-size 256 \
  --device cuda:0 \
  --patience 45 \
  --pca-dim 128 \
  --wandb-project pd-hand-vjepa2 \
  --wandb-run-name item3_4_fold0_vjepa21_nocrop_mean_pca128_sweep \
  --wandb-mode online
```

Observed fold-0 validation results from frozen V-JEPA 2.1 cached embeddings:

| Visual cache | Best val accuracy | Best count | Notes |
| --- | ---: | ---: | --- |
| hand-crop mean | 53.0% | 35/66 | PCA-128 temporal heads |
| hand-crop mean+std | 50.0% | 33/66 | PCA-128 temporal heads |
| no-crop mean | 56.1% | 37/66 | PCA-128 temporal heads |
| hand-crop + no-crop concat | 54.5% | 36/66 | PCA-128 temporal heads |
| cross-run probability ensemble | 57.6% | 38/66 | Validation-selected ensemble across saved pure-vision heads |

The target `> 70%` requires at least `47/66` on this validation split. These frozen V-JEPA 2.1 cached-feature runs did not reach that target.

## Trainable Pure-Vision V-JEPA 2.1 Runs

The video eval path now supports opt-in encoder fine-tuning:

```yaml
experiment:
  optimization:
    finetune_encoder: true
    encoder_trainable_blocks: 1
    encoder_trainable_patterns:
    - model.norm
    - model.fc_norm
    encoder_lr: 0.000003
```

Defaults remain frozen. When fine-tuning is enabled, the training loop wraps the encoder in DDP, includes trainable encoder parameters in AdamW, removes `torch.no_grad()` only for the trainable training forward, and saves only trainable encoder weights in checkpoints.

The new `head_type: temporal_stats` is a pure-vision head. It reshapes V-JEPA tokens into `T x spatial_tokens`, spatially pools each tubelet, then applies the same temporal summary pattern used by the cached V-JEPA temporal-encoder sweep.

Local 4-H100 tests on May 20, 2026 used:

```bash
OMP_NUM_THREADS=2 /home/yl2428/.conda/envs/video-llama/bin/python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=<port> \
  -m evals.main --debugmode true --use_fsdp \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  --fname <config>
```

Observed pure-vision trainable results so far:

| Config | Trainable encoder | Batch/GPU | Best val accuracy | Coverage | Notes |
| --- | --- | ---: | ---: | --- | --- |
| `stats_nocrop_lastblock_ft_4seg` | last block | 1 | 28.8% | 43.8% / 100.0% | Smoke run; low raw-frame coverage |
| `stats_nocrop_lastblock_ft_12seg_b2` | last block | 2 | 34.4% | 94.0% / 98.3% | Collapsed to class 2 |
| `temporalstats_nocrop_lastblock_ft_12seg_b2` | last block | 2 | 34.4% | 94.0% / 98.3% | Collapsed to class 2 then class 1 |
| `temporalstats_balanced_nocrop_lastblock_ft_12seg_b2` | last block | 2 | 34.4% | 94.0% / 98.3% | Balanced loss shifted collapse across classes |
| `temporalstats_nocrop_last4_ft_12seg_b1` | last 4 blocks | 1 | 30.9% | 93.9% / 98.2% | 50.4M trainable params; still collapsed |

Memory was not the limiter: last-block 12-segment batch-2 used about `18.8 GB` per H100, and last-4-block 12-segment batch-1 used about `24.3 GB` per H100. These runs did not reach the pure-vision `>=70%` target.

## V-JEPA 2.1 LoRA JEPA Adaptation

The domain-adaptive JEPA path now supports LoRA in the V-JEPA 2.1 encoder. The first implementation injects LoRA into encoder attention `attn.qkv` and `attn.proj` linears, freezes the base encoder, keeps the JEPA predictor trainable, and copies the LoRA-equipped encoder into the EMA target encoder. This is the intended first pure-vision adaptation experiment because it learns from PD hand-task clips without optimizing the small/noisy 0-4 labels.

Implementation files:

- `src/utils/lora.py`
- `app/vjepa_2_1/utils.py`
- `app/vjepa_2_1/train.py`
- `evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_vjepa21.py`

Training config:

```bash
configs/train_2_1/pd_hand/vitl384-lora-jepa-adapt-32f-step1-fold0.yaml
```

Slurm run:

```bash
sbatch scripts/pd_hand/run_item_3_4_fold0_vjepa21_lora_jepa_adapt_wandb_ddp4.sbatch
```

Interactive 4-H100 run:

```bash
salloc --partition=gpu --nodes=1 --ntasks=4 --gres=gpu:h100:4 \
  --cpus-per-task=8 --mem=320G --time=12:00:00
bash scripts/pd_hand/run_item_3_4_fold0_vjepa21_lora_jepa_adapt_wandb_ddp4.sbatch
```

The LoRA config uses only `fold_0_train.csv` for JEPA adaptation. Do not adapt on `fold_0_val.csv` unless explicitly reporting a transductive/leaky experiment. The saved LoRA checkpoint can be evaluated with the existing V-JEPA 2.1 frozen wrapper because LoRA weights are merged into normal Linear weights at load time.

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

For the stats-pooling balanced no-augmentation/no-crop run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-stats-balanced-noaug-nocrop \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_stats_balanced_noaug_nocrop_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-stats-balanced-noaug-nocrop \
JOB_NAME=vjepa_i34_f0_stbnc21 \
LOG_PREFIX=acc_vjepa21_dense_stats_balanced_noaug_nocrop_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862039
```

For the stats-pooling metadata balanced no-augmentation/no-crop run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-balanced-noaug-nocrop \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_statsmeta_balanced_noaug_nocrop_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-balanced-noaug-nocrop \
JOB_NAME=vjepa_i34_f0_stmnc21 \
LOG_PREFIX=acc_vjepa21_dense_statsmeta_balanced_noaug_nocrop_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862041
```

For the stats-pooling metadata expected-round run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-eround-balanced-noaug-nocrop \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_statsmeta_eround_balanced_noaug_nocrop_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-eround-balanced-noaug-nocrop \
JOB_NAME=vjepa_i34_f0_stmenc21 \
LOG_PREFIX=acc_vjepa21_dense_statsmeta_eround_balanced_noaug_nocrop_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862042
```

For the scaled-metadata stats-pooling run:

```bash
TAG=pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-scaled-noaug-nocrop \
RUN_ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_accuracy_vjepa21_dense_statsmeta_scaled_noaug_nocrop_h100x2/video_classification_frozen/pd-hand-item-3_4-fold-0-acc-vjepa21-dense-statsmeta-scaled-noaug-nocrop \
JOB_NAME=vjepa_i34_f0_stmsnc21 \
LOG_PREFIX=acc_vjepa21_dense_statsmeta_scaled_noaug_nocrop_ddp2 \
scripts/pd_hand/monitor_item_3_4_fold0.sh 28862043
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
- Dense balanced-softmax no-smoothing comparison job `28862033` kept `class_weight: balanced` but removed label smoothing to test direct raw-accuracy optimization. Fold-0 train has no class-4 examples, so the auto class-4 training weight was `0.0`. Epoch 1 reached validation accuracy `36.36364`, Spearman `0.31613`, QWK `0.14544`, MAE `0.86364`, train coverage `0.93951/0.97561`, and val coverage `0.94125/0.98255`. It was stopped during epoch 2 because train accuracy stayed weak and the queued stats-pooling run needed GPUs.
- Dense softmax no-augmentation comparison job `28862034` was stopped after the bf16 fix because it had started under the older fp16 autocast path.
- Replacement dense softmax no-augmentation job `28862035` kept the dense 32-frame, stride-1, 12-segment coverage settings while disabling training RandAugment/random erasing and constraining random resized crop to `0.9-1.0` square crops. It reached epoch 1 validation accuracy `30.30303`, Spearman `0.11738`, QWK `0.0`, and MAE `0.78788`; it was stopped to free 2 H100s.
- Future runs after the bf16 fix use true `torch.bfloat16` autocast in the video eval path. Earlier jobs used the existing video-eval behavior, where `use_bfloat16: true` actually selected fp16 autocast.
- No-MediaPipe-crop jobs `28862036` and `28862037` were stopped because they inherited the base no-augmentation W&B/config values through the shared runner.
- Dedicated no-crop softmax job `28862038` confirmed `dataset_kwargs: {hand_crop: false}` plus W&B run id `pd-hand-item-3_4-fold-0-acc-vjepa21-dense-softmax-noaug-nocrop` in startup, then reached epoch 1 validation accuracy `30.30303`, Spearman `nan`, QWK `0.0`, and MAE `0.78788`; it was stopped.
- A stats-pooling probe head is implemented as `head_type: stats`. It pools frozen V-JEPA token features with mean/std/max and uses a small MLP, giving a lower-variance alternative to the single-query attentive probe for this small fold.
- Stats-pooling balanced no-augmentation/no-crop job `28862039` failed during validation after live code was patched under the running job; the dataset path is now backward-compatible for pre-patch `ClipDataset` instances.
- Stats-pooling metadata job `28862041` appended manifest metadata one-hots for `side` (`Left`, `Right`) and `dx` (`HC`, `NDC`, `PD`, `PPD`) to the mean/std/max pooled frozen-token vector. Fold 0 train has all six metadata categories; fold 0 validation lacks `PPD`, so that feature is always zero in validation. It reached best validation accuracy `34.84848` by epoch 2, then plateaued; it was stopped after epoch 4.
- Stats-pooling metadata epoch 1 reached validation accuracy `33.33333`, Spearman `0.49243`, QWK `0.0`, and MAE `0.96970`, but the selected classifier predicted every validation clip as class 2. Epoch 4 had QWK `0.19171` but only `33.33333` accuracy with predictions limited to classes 0/1.
- Stats-pooling metadata expected-round job `28862042` reached epoch 1 validation accuracy `30.30303`, Spearman `0.50118`, QWK `0.0`, and MAE `0.78788`; it was stopped.
- Scaled-metadata stats job `28862043` multiplied metadata one-hots by `32.0` before concatenation so the six metadata inputs were not drowned out by the `3072` pooled video-feature dimensions. Startup confirmed W&B online logging, strict V-JEPA 2.1 checkpoint loading, and `metadata_scale: 32.0`. It was stopped at epoch 6 because it was still far below target: best validation accuracy `39.39394`, Spearman `0.37536`, QWK `0.08662`, and MAE `0.83333`.
- The best verified non-V-JEPA baseline so far is the CPU MediaPipe/kinematic baseline. With stride-2 cached features and a validation-selected median-vote ensemble, it reached validation accuracy `72.72727` (`48/66`), QWK `0.81446`, and MAE `0.28788`. This clears the active `>=70%` fold-0 validation target.
- The ensemble is named `validation_selected_median_vote_v1` in `scripts/pd_hand/train_finger_tapping_kinematic_baseline.py`. It votes over `old_118:extra_trees_old_seed0`, `meta:extra_trees_old_seed0`, `old_openmeta:random_forest`, and `meta:gradient_boosting`.
- This baseline includes optional same-visit item 3.5 labels from the manifest as context, and the ensemble recipe was selected on fold-0 validation predictions after model-family search. Treat it as a diagnostic/tuned fold-0 result, not as a pure one-clip-in model or an untouched generalization estimate.
- The previous best single kinematic model was `old_118:extra_trees_old_seed98`, with validation accuracy `68.18182` (`45/66`), QWK `0.81255`, and MAE `0.31818`.
- The verified ensemble confusion matrix is:

```text
[[16, 2, 1, 0, 0],
 [ 3,14, 3, 0, 0],
 [ 0, 5,17, 0, 0],
 [ 0, 0, 3, 1, 0],
 [ 0, 0, 0, 1, 0]]
```

The result path is:

```text
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/results_stride2.json
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/saved_prediction_vote_ensembles_stride2.json
```

Additional search artifacts:

```text
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/quick_tuning_stride2.json
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/old_feature_extra_trees_search_stride2.json
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/stacked_oof_stride2.json
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/kinematic_baselines/item_3_4/fold_0/validation_calibrated_fusion_stride2.json
```

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
