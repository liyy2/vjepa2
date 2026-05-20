#!/bin/bash
#SBATCH --job-name=vjepa_i34_f0_sp3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:3
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0/run_logs/spearman_ddp3_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0/run_logs/spearman_ddp3_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONFIG="${CONFIG:-${REPO}/configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl_ssv2_corn_spearman.yaml}"
RUN_FOLDER="${RUN_FOLDER:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_spearman_h100x3}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO}/wandb_api.txt}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
export REPO CONFIG RUN_FOLDER WANDB_API_KEY_FILE CONDA CONDA_ENV

mkdir -p "$RUN_FOLDER"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export WANDB_MODE=online
unset WANDB_API_KEY
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))

cd "$REPO"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_JOB_NODELIST}"
echo "ntasks=${SLURM_NTASKS}"
echo "config=${CONFIG}"
echo "run_folder=${RUN_FOLDER}"
nvidia-smi

"$REPO/scripts/pd_hand/prepare_wandb_auth.sh"

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU

srun --ntasks="${SLURM_NTASKS}" --kill-on-bad-exit=1 /bin/bash -lc '
set -euo pipefail
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
unset WANDB_API_KEY
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"

DEVICES=()
for ((idx=0; idx<SLURM_NTASKS; idx++)); do
  DEVICES+=("cuda:${idx}")
done

echo "rank=${RANK} local_rank=${LOCAL_RANK} host=$(hostname) cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m evals.main \
  --debugmode true \
  --use_fsdp \
  --devices "${DEVICES[@]}" \
  --folder "$RUN_FOLDER" \
  --override_config_folder \
  --fname "$CONFIG"
'
