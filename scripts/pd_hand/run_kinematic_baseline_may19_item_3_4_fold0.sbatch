#!/bin/bash
#SBATCH --job-name=kin_may19_i34f0
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/kinematic_baselines/item_3_4/fold_0/run_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/kinematic_baselines/item_3_4/fold_0/run_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
BASE=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
OUT=${BASE}/kinematic_baselines/item_3_4/fold_0
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama

mkdir -p "$OUT"
cd "$REPO"

echo "job=${SLURM_JOB_ID} start=$(date -Iseconds)"

"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/train_finger_tapping_kinematic_baseline.py \
    --train-csv    "${BASE}/splits/item_3_4/fold_0_train.csv" \
    --val-csv      "${BASE}/splits/item_3_4/fold_0_val.csv" \
    --manifest-csv "${BASE}/hand_clip_manifest.csv" \
    --out-dir      "${OUT}" \
    --frame-stride 2 \
    --num-workers  14

echo "job=${SLURM_JOB_ID} end=$(date -Iseconds)"
