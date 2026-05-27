#!/bin/bash
#SBATCH --job-name=may24-fair-stack
#SBATCH --partition=day
#SBATCH --array=0-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/may24_fair_stack_%A_%a.out
#SBATCH --error=logs/may24_fair_stack_%A_%a.err

set -euo pipefail

cd /gpfs/milgram/pi/scherzer/yl2428/vjepa2
mkdir -p logs

ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
FOLD=${SLURM_ARRAY_TASK_ID}

CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda

"${CONDA}" run --no-capture-output -n video-llama \
  python scripts/pd_hand/hybrid_multiseed.py \
    --train-csv "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/train_features_with_oof.csv" \
    --val-csv "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/val_features_with_oof.csv" \
    --out-json "${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/multiseed_oof_probs_kinonly.json" \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --n-estimators 1000
