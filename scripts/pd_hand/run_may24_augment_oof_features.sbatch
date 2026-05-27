#!/bin/bash
#SBATCH --job-name=may24-oof-join
#SBATCH --partition=day
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/may24_oof_join_%A_%a.out
#SBATCH --error=logs/may24_oof_join_%A_%a.err

set -euo pipefail

cd /gpfs/milgram/pi/scherzer/yl2428/vjepa2
mkdir -p logs

ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
FOLD=${SLURM_ARRAY_TASK_ID}

CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda

"${CONDA}" run --no-capture-output -n video-llama \
  python scripts/pd_hand/augment_with_oof_probs.py \
    --oof-npz "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_loraft_latest_clean.npz" \
    --fold-train-csv "${ROOT}/splits/item_3_4/fold_${FOLD}_train.csv" \
    --fold-val-csv "${ROOT}/splits/item_3_4/fold_${FOLD}_val.csv" \
    --kin-train-csv "${ROOT}/kinematic_baselines/item_3_4/fold_${FOLD}/train_features_stride2.csv" \
    --kin-val-csv "${ROOT}/kinematic_baselines/item_3_4/fold_${FOLD}/val_features_stride2.csv" \
    --out-train-csv "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/train_features_with_oof.csv" \
    --out-val-csv "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/val_features_with_oof.csv"
