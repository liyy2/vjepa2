#!/bin/bash
#SBATCH --job-name=loraft_aggr_may19_i34f0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive/run_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive/run_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
CONFIG=${REPO}/configs/eval_2_1/pd_hand_may19_item_3_4_fold0_supervised_lora_ft_aggressive.yaml
RUN_FOLDER=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama

mkdir -p "$RUN_FOLDER"
cd "$REPO"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
unset WANDB_API_KEY || true
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

echo "job=${SLURM_JOB_ID} start=$(date -Iseconds)"
nvidia-smi -L || true

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

"$CONDA" run --no-capture-output -n "$ENV" python -m evals.main \
  --devices cuda:0 \
  --folder "$RUN_FOLDER" \
  --override_config_folder \
  --fname "$CONFIG"

echo "job=${SLURM_JOB_ID} end=$(date -Iseconds)"
