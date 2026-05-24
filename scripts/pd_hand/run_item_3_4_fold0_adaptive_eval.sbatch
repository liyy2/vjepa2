#!/bin/bash
#SBATCH --job-name=vjepa_i34_adapt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_adaptive/item_3_4/run_logs/adaptive_eval_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_adaptive/item_3_4/run_logs/adaptive_eval_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/eval_2_1/pd_hand_item_3_4_fold0_adaptive_frozen_statsmlp.yaml}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO}/wandb_api.txt}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
LOG_ROOT="/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_adaptive/item_3_4/run_logs"
export REPO CONFIG WANDB_API_KEY_FILE CONDA CONDA_ENV

mkdir -p "$LOG_ROOT"

export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export WANDB_CONFIG_DIR=/home/yl2428/.config/wandb
export WANDB_CACHE_DIR=/home/yl2428/.cache/wandb
export WANDB_DATA_DIR=/home/yl2428/.local/share/wandb
mkdir -p "$XDG_CACHE_HOME" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export WANDB_MODE=online
export WANDB_ENTITY=yl2428
unset WANDB_API_KEY || true
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "$REPO"

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "config=${CONFIG}"
nvidia-smi

"$REPO/scripts/pd_hand/prepare_wandb_auth.sh"

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m evals.main \
  --debugmode true \
  --devices cuda:0 \
  --fname "$CONFIG"
