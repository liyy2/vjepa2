#!/bin/bash
#SBATCH --job-name=may24_cache_vjepa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/may24_cache_fold%a_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/may24_cache_fold%a_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
ROOT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
FOLD=${SLURM_ARRAY_TASK_ID}

CFG=${REPO}/configs/eval_2_1/pd_hand_may24_vitl384_lora_cache.yaml
SPLITS=${ROOT}/splits/item_3_4
CHECKPOINT=${ROOT}/vjepa2_evals/item_3_4/fold_${FOLD}_supervised_lora_ft_aggressive/merged_latest_for_cache.pt
OUT=${ROOT}/vjepa2_embeddings/item_3_4/fold_${FOLD}_loraft_aggressive_latest_adaptive

mkdir -p "$OUT"
cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
  --config "$CFG" \
  --dataset-train "$SPLITS/fold_${FOLD}_train.csv" \
  --dataset-val "$SPLITS/fold_${FOLD}_val.csv" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT" \
  --split both \
  --batch-size 1 \
  --num-workers 6 \
  --device cuda:0 \
  --hand-crop \
  --crop-size 256 \
  --crop-scale 2.35 \
  --pool mean \
  --adaptive-num-clips \
  --adaptive-min-clips 4 \
  --adaptive-max-clips 48
