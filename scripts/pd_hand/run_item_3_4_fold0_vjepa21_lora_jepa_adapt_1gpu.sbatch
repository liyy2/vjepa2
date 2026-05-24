#!/bin/bash
#SBATCH --job-name=vjepa_i34_f0_lora1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/run_logs/lora_jepa_adapt_1gpu_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/run_logs/lora_jepa_adapt_1gpu_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/train_2_1/pd_hand/vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows-1gpu.yaml}"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import os
import yaml

with open(os.environ["CONFIG"], "r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
print("config_folder=" + str(cfg["folder"]))
print("data_num_workers=" + str(cfg["data"].get("num_workers")))
print("adaptive_num_clips=" + str(cfg["data"].get("dataset_kwargs", {}).get("adaptive_num_clips")))
print("clip_microbatch_size=" + str(cfg["optimization"].get("clip_microbatch_size")))
print("wandb_enabled=" + str(cfg.get("wandb", {}).get("enabled")))
PY
nvidia-smi

"$REPO/scripts/pd_hand/prepare_wandb_auth.sh"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU

srun --ntasks="${SLURM_NTASKS}" \
  --export=ALL,REPO="${REPO}",CONFIG="${CONFIG}",CONDA="${CONDA}",CONDA_ENV="${CONDA_ENV}" \
  --kill-on-bad-exit=1 /bin/bash -lc '
set -euo pipefail
REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/train_2_1/pd_hand/vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows-1gpu.yaml}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
export REPO CONFIG CONDA CONDA_ENV
cd "$REPO"
export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export WANDB_CONFIG_DIR=/home/yl2428/.config/wandb
export WANDB_CACHE_DIR=/home/yl2428/.cache/wandb
export WANDB_DATA_DIR=/home/yl2428/.local/share/wandb
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2
export WANDB_MODE=online
export WANDB_ENTITY=yl2428
unset WANDB_API_KEY
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

echo "rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -c "import os, yaml; from app.scaffold import main as app_main; params = yaml.load(open(os.environ[\"CONFIG\"]), Loader=yaml.FullLoader); app_main(params[\"app\"], args=params)"
'
