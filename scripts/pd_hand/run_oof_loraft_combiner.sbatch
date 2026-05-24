#!/bin/bash
#SBATCH --job-name=oof_loraft_comb
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_0/oof_loraft_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_0/oof_loraft_%j.err

set -euo pipefail
REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
cd "$REPO"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

CACHE=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive
FOLDS=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/splits/item_3_4
OUT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/hybrid/item_3_4/fold_0/oof_loraft.npz

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/oof_vjepa_combiner_predictions.py \
  --cache-train "$CACHE/vjepa21_vitl384_train_32f_step1_adaptive_max48_handcrop_cs256_sc2.35_mean.npz" \
  --cache-val   "$CACHE/vjepa21_vitl384_val_32f_step1_adaptive_max48_handcrop_cs256_sc2.35_mean.npz" \
  --fold-train-csv "$FOLDS/fold_0_train.csv" \
  --fold-val-csv   "$FOLDS/fold_0_val.csv" \
  --out-npz "$OUT" \
  --num-inner-folds 5 \
  --epochs 150 \
  --patience 80 \
  --d-model 128 --n-layers 2 --n-heads 4 \
  --num-segments-for-pos 48 --tubelets-per-segment 16 \
  --seed 0
