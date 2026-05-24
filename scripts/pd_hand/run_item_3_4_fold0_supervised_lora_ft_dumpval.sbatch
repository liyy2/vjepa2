#!/bin/bash
#SBATCH --job-name=vjepa_dumpval_i34_f0
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_val_dump/run_logs/dumpval_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_val_dump/run_logs/dumpval_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/eval_2_1/pd_hand_item_3_4_fold0_supervised_lora_ft_dumpval.yaml}"
RUN_FOLDER="${RUN_FOLDER:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_val_dump}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
export REPO CONFIG RUN_FOLDER CONDA CONDA_ENV

mkdir -p "$RUN_FOLDER"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
unset WANDB_API_KEY || true
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

cd "$REPO"

echo "job_id=${SLURM_JOB_ID}"
echo "config=${CONFIG}"
echo "run_folder=${RUN_FOLDER}"
nvidia-smi

unset SLURM_MEM_PER_CPU || true
unset SLURM_MEM_PER_GPU || true

export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m evals.main \
  --debugmode true \
  --val_only \
  --devices cuda:0 \
  --folder "$RUN_FOLDER" \
  --override_config_folder \
  --fname "$CONFIG"
