#!/bin/bash
#SBATCH --job-name=comb_loraft_aggr_adapt_long_may19
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/combiner/item_3_4/fold_0_loraft_aggressive_adaptive_long/run_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/combiner/item_3_4/fold_0_loraft_aggressive_adaptive_long/run_%j.err

set -euo pipefail

REPO=/gpfs/milgram/pi/scherzer/yl2428/vjepa2
CACHE=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4/fold_0_loraft_aggressive_adaptive
OUT_BASE=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19/combiner/item_3_4/fold_0_loraft_aggressive_adaptive_long
CONDA=/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda
ENV=video-llama
mkdir -p "$OUT_BASE"; cd "$REPO"
export PYTHONUNBUFFERED=1; export WANDB_MODE=disabled

TRAIN_NPZ=$(ls "$CACHE"/vjepa21_vitl384_train_32f_step1_adaptive*_mean.npz 2>/dev/null | head -1)
VAL_NPZ=$(ls   "$CACHE"/vjepa21_vitl384_val_32f_step1_adaptive*_mean.npz   2>/dev/null | head -1)
if [ -z "$TRAIN_NPZ" ] || [ -z "$VAL_NPZ" ]; then echo "missing cache"; exit 1; fi

run () {
  local tag=$1; shift
  local outdir="${OUT_BASE}/${tag}"
  mkdir -p "$outdir"
  echo "=== ${tag} ==="
  "$CONDA" run --no-capture-output -n "$ENV" python scripts/pd_hand/train_corn_combiner.py \
    --out-dir "$outdir" "$@"
}

COMMON_TIGHT="--epochs 1500 --batch-size 16 --warmup-epochs 20 --weight-decay 0.1 \
              --class-weight balanced --select-by qwk_acc \
              --patience 900 --cosine-epochs 400 --final-lr-ratio 0.01 --device cuda:0 --wandb-mode disabled \
              --num-segments-for-pos 48 --tubelets-per-segment 16"

# Same configs as the frozen-adaptive long sweep, but on aggressive-LoRA-FT adaptive cache
for seed in 0 1 2 17 42 98; do
  run "tiny_d128_l2_seed${seed}" \
    --train-npz "$TRAIN_NPZ" --val-npz "$VAL_NPZ" \
    --scale-names loraft_adaptive \
    --d-model 128 --n-layers 2 --n-heads 4 --seed $seed \
    --dropout 0.4 --lr 3e-4 \
    --mixup-alpha 0.2 --segment-dropout 0.2 --token-dropout 0.1 \
    ${COMMON_TIGHT}
done

for seed in 0 1 2 17; do
  run "med_d192_l3_seed${seed}" \
    --train-npz "$TRAIN_NPZ" --val-npz "$VAL_NPZ" \
    --scale-names loraft_adaptive \
    --d-model 192 --n-layers 3 --n-heads 6 --seed $seed \
    --dropout 0.35 --lr 3e-4 \
    --mixup-alpha 0.2 --segment-dropout 0.15 --token-dropout 0.05 \
    ${COMMON_TIGHT}
done

for seed in 0 1; do
  run "deep_d192_l6_seed${seed}" \
    --train-npz "$TRAIN_NPZ" --val-npz "$VAL_NPZ" \
    --scale-names loraft_adaptive \
    --d-model 192 --n-layers 6 --n-heads 6 --seed $seed \
    --dropout 0.45 --lr 2e-4 \
    --mixup-alpha 0.3 --segment-dropout 0.2 --token-dropout 0.1 \
    ${COMMON_TIGHT}
done

echo "loraft aggressive adaptive long sweep done at $(date -Iseconds)"
