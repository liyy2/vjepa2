#!/bin/bash
#SBATCH --job-name=merge_aggr_may19_i34f0
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive/merge_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive/merge_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
RUN_FOLDER=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_evals/item_3_4/fold_0_supervised_lora_ft_aggressive
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
cd "$REPO"

NESTED=$(find "$RUN_FOLDER"/video_classification_frozen -maxdepth 2 -type d 2>/dev/null | tail -1)
CKPT=""
for cand in "$NESTED/best.pt" "$NESTED/latest.pt"; do
  if [ -f "$cand" ]; then CKPT="$cand"; break; fi
done
if [ -z "$CKPT" ]; then
  echo "No checkpoint found"; ls -laR "$RUN_FOLDER" >&2; exit 1
fi
echo "merging $CKPT"

OUT=${RUN_FOLDER}/merged_for_cache.pt
"$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/merge_lora_from_base.py \
  --base /gpfs/milgram/pi/scherzer/yl2428/vjepa2/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt \
  --lora "$CKPT" --out "$OUT" \
  --lora-alpha 32 --lora-rank 16

ls -la "$OUT" || true
echo "merge done at $(date -Iseconds)"
