#!/bin/bash
#SBATCH --job-name=may24_oof_vjepa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_%a/oof_clean_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_%a/oof_clean_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
FOLD=${SLURM_ARRAY_TASK_ID}

CACHE=${ROOT}/vjepa2_embeddings/item_3_4/fold_${FOLD}_loraft_aggressive_latest_adaptive
SPLITS=${ROOT}/splits/item_3_4
OUT=${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_loraft_latest_clean.npz

mkdir -p "$(dirname "$OUT")"
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/oof_vjepa_combiner_predictions.py \
  --cache-train "$CACHE/vjepa21_vitl384_train_32f_step1_adaptive_max48_handcrop_cs256_sc2.35_mean.npz" \
  --cache-val "$CACHE/vjepa21_vitl384_val_32f_step1_adaptive_max48_handcrop_cs256_sc2.35_mean.npz" \
  --fold-train-csv "$SPLITS/fold_${FOLD}_train.csv" \
  --fold-val-csv "$SPLITS/fold_${FOLD}_val.csv" \
  --out-npz "$OUT" \
  --num-inner-folds 5 \
  --epochs 150 \
  --patience 80 \
  --d-model 128 \
  --n-layers 2 \
  --n-heads 4 \
  --num-segments-for-pos 48 \
  --tubelets-per-segment 16 \
  --seed 0 \
  --no-val-tuning
