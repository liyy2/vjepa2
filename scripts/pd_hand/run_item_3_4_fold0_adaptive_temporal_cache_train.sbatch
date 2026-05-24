#!/bin/bash
#SBATCH --job-name=i34_adapt_tstats
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/run_logs/adaptive_tstats_%j.out
#SBATCH --error=/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/run_logs/adaptive_tstats_%j.err

set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
PYTHON="${PYTHON:-/home/yl2428/.conda/envs/video-llama/bin/python}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${REPO}/configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_lora_e20_attentive.yaml}"
OUT_EMB="${OUT_EMB:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0_lora_handcrop_e20_adaptive_topkpool}"
OUT_SWEEP="${OUT_SWEEP:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/fold_0_lora_handcrop_e20_adaptive_topkpool}"
LOG_ROOT="/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_temporal_encoder/item_3_4/run_logs"

export HOME=/home/yl2428
export XDG_CACHE_HOME=/home/yl2428/.cache
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled
mkdir -p "$OUT_EMB" "$OUT_SWEEP" "$LOG_ROOT"

cd "$REPO"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "out_emb=${OUT_EMB}"
echo "out_sweep=${OUT_SWEEP}"
nvidia-smi

TMPCFG="${OUT_EMB}/cache_config_adaptive_lora_e20.yaml"
"$PYTHON" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("${CONFIG_TEMPLATE}").read())
cfg["model_kwargs"]["wrapper_kwargs"]["clip_batch_size"] = 2
Path("${TMPCFG}").write_text(yaml.dump(cfg, sort_keys=False))
print("wrote", "${TMPCFG}")
PY

for CROP_NAME in nocrop handcrop; do
  CROP_ARGS=()
  if [[ "$CROP_NAME" == "handcrop" ]]; then
    CROP_ARGS=(--hand-crop --crop-size 128 --crop-scale 1.5)
  fi
  LOG="${OUT_EMB}/cache_${CROP_NAME}_adaptive_topk.log"
  echo "cache ${CROP_NAME}: ${LOG}"
  "$PYTHON" -u scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
    --config "$TMPCFG" \
    --out-dir "$OUT_EMB" \
    --split both \
    --batch-size 1 \
    --num-workers 2 \
    --pool topk_mean \
    --topk 16 \
    --adaptive-num-clips \
    --adaptive-duration-col duration_s \
    --adaptive-min-clips 1 \
    --device cuda:0 \
    --force \
    "${CROP_ARGS[@]}" \
    > "$LOG" 2>&1
done

"$PYTHON" - <<PY
import json
import numpy as np
from pathlib import Path
root = Path("${OUT_EMB}")
for split in ["train", "val"]:
    hc = np.load(root / f"vjepa21_vitl384_{split}_32f_step1_adaptive_handcrop_cs128_sc1.5_topk_mean.npz")
    nc = np.load(root / f"vjepa21_vitl384_{split}_32f_step1_adaptive_nocrop_topk_mean.npz")
    if not np.array_equal(hc["y"], nc["y"]):
        raise SystemExit(f"{split}: labels differ between handcrop/nocrop")
    if not np.array_equal(hc["mask"], nc["mask"]):
        raise SystemExit(f"{split}: masks differ between handcrop/nocrop")
    out = root / f"vjepa21_vitl384_{split}_32f_step1_adaptive_handcrop_nocrop_topk_mean_concat.npz"
    cfg = {
        "sources": [
            str(root / f"vjepa21_vitl384_{split}_32f_step1_adaptive_handcrop_cs128_sc1.5_topk_mean.npz"),
            str(root / f"vjepa21_vitl384_{split}_32f_step1_adaptive_nocrop_topk_mean.npz"),
        ],
        "kind": "adaptive_handcrop_nocrop_topk_concat",
    }
    np.savez_compressed(
        out,
        x=np.concatenate([hc["x"].astype(np.float16), nc["x"].astype(np.float16)], axis=-1),
        y=hc["y"],
        mask=hc["mask"],
        coverage=hc["coverage"],
        config=json.dumps(cfg),
        temporal_length=np.asarray([hc["x"].shape[1]], dtype=np.int64),
        input_dim=np.asarray([hc["x"].shape[2] + nc["x"].shape[2]], dtype=np.int64),
    )
    print(f"wrote {out} x={np.load(out)['x'].shape} valid_tokens={hc['mask'].sum(axis=1).min()}/{np.median(hc['mask'].sum(axis=1))}/{hc['mask'].sum(axis=1).max()}")
PY

TRAIN_NPZ="${OUT_EMB}/vjepa21_vitl384_train_32f_step1_adaptive_handcrop_nocrop_topk_mean_concat.npz"
VAL_NPZ="${OUT_EMB}/vjepa21_vitl384_val_32f_step1_adaptive_handcrop_nocrop_topk_mean_concat.npz"

for GRID in velocity mega_seeds; do
  OUT_DIR="${OUT_SWEEP}/${GRID}"
  mkdir -p "$OUT_DIR"
  echo "train temporal-stats grid=${GRID} out=${OUT_DIR}"
  "$PYTHON" -u scripts/pd_hand/train_temporal_encoder.py \
    --train-npz "$TRAIN_NPZ" \
    --val-npz "$VAL_NPZ" \
    --out-dir "$OUT_DIR" \
    --grid "$GRID" \
    --rank-by qwk \
    --epochs 80 \
    --patience 25 \
    --batch-size 16 \
    --pca-dim 128 \
    --segment-length 16 \
    --device cuda:0
done

"$PYTHON" -u scripts/pd_hand/cross_sweep_ensemble.py \
  --sweeps "${OUT_SWEEP}/velocity" "${OUT_SWEEP}/mega_seeds" \
  --max-K 100 \
  --out "${OUT_SWEEP}/cross_sweep_ensemble.json"
