#!/bin/bash
#SBATCH --job-name=vjepa_i34_f0_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=320G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/run_logs/lora_jepa_adapt_ddp4_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/run_logs/lora_jepa_adapt_ddp4_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/train_2_1/pd_hand/vitl384-lora-jepa-adapt-32f-step1-fold0.yaml}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO}/wandb_api.txt}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
LOG_ROOT="/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/run_logs"
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
export OMP_NUM_THREADS=2
export WANDB_MODE=online
export WANDB_ENTITY=yl2428
unset WANDB_API_KEY
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

cd "$REPO"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "ntasks=${SLURM_NTASKS}"
echo "config=${CONFIG}"
nvidia-smi

"$REPO/scripts/pd_hand/prepare_wandb_auth.sh"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU

srun --ntasks="${SLURM_NTASKS}" --kill-on-bad-exit=1 /bin/bash -lc '
set -euo pipefail
cd "$REPO"
export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export WANDB_CONFIG_DIR=/home/yl2428/.config/wandb
export WANDB_CACHE_DIR=/home/yl2428/.cache/wandb
export WANDB_DATA_DIR=/home/yl2428/.local/share/wandb
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export WANDB_MODE=online
export WANDB_ENTITY=yl2428
unset WANDB_API_KEY
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -c "import os, yaml; from app.scaffold import main as app_main; params = yaml.load(open(os.environ[\"CONFIG\"]), Loader=yaml.FullLoader); app_main(params[\"app\"], args=params)"
'
