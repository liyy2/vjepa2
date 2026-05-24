#!/bin/bash
#SBATCH --job-name=tstats_loraft_aggr_adapt_may19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/temporal_stats/item_3_4/fold_0_loraft_aggressive_adaptive/run_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/temporal_stats/item_3_4/fold_0_loraft_aggressive_adaptive/run_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
CACHE=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive
OUT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/temporal_stats/item_3_4/fold_0_loraft_aggressive_adaptive
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama

mkdir -p "$OUT"; cd "$REPO"
export PYTHONUNBUFFERED=1; export WANDB_MODE=disabled

TRAIN_NPZ=$(ls "$CACHE"/vjepa21_vitl384_train_32f_step1_adaptive*_mean.npz 2>/dev/null | head -1)
VAL_NPZ=$(ls   "$CACHE"/vjepa21_vitl384_val_32f_step1_adaptive*_mean.npz   2>/dev/null | head -1)
if [ -z "$TRAIN_NPZ" ] || [ -z "$VAL_NPZ" ]; then echo "missing cache"; exit 1; fi

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/train_temporal_encoder.py \
  --train-npz "$TRAIN_NPZ" --val-npz "$VAL_NPZ" \
  --out-dir "$OUT/velocity_full" \
  --grid velocity --rank-by qwk \
  --epochs 120 --batch-size 16 --device cuda:0 \
  --pca-dim 128 --patience 40 \
  --sample-balanced-normalization \
  --pca-tokens-per-sample 16 \
  --wandb-mode disabled

echo "done at $(date -Iseconds)"
