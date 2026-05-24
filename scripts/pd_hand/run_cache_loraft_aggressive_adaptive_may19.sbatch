#!/bin/bash
#SBATCH --job-name=cachev_loraft_aggr_adapt_may19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive/run_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive/run_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
CFG=${REPO}/configs/eval_2_1/pd_hand_may19_item_3_4_fold0_vjepa2_1_vitl384_corn_loraft_aggressive_adaptive.yaml
OUT=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
mkdir -p "$OUT"; cd "$REPO"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job=${SLURM_JOB_ID} start=$(date -Iseconds)"
nvidia-smi -L || true

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
    --config "$CFG" --out-dir "$OUT" --split both \
    --batch-size 1 --num-workers 6 --device cuda:0 \
    --hand-crop --crop-size 256 --crop-scale 2.35 --pool mean \
    --adaptive-num-clips --adaptive-min-clips 4 --adaptive-max-clips 48

echo "job=${SLURM_JOB_ID} end=$(date -Iseconds)"
