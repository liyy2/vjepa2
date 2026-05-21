#!/bin/bash
# Cache V-JEPA 2.1 features using a LoRA-adapted encoder.
#
# Usage:
#   bash scripts/pd_hand/cache_with_lora_encoder.sh \\
#       <lora_ckpt_path> <out_dir>
#
# Workflow:
#   1) merge LoRA into base weights -> merged_for_cache.pt
#   2) write a temp cache config that points the encoder at the merged checkpoint
#   3) call cache_vjepa21_temporal_embeddings.py once per (split × hand_crop)

set -euo pipefail

LORA_CKPT="${1:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_jepa_lora_adapt/item_3_4/fold_0_vitl384_32f_step1_rank8/latest.pth.tar}"
OUT_DIR="${2:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0_lora_rank8}"
REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
PYTHON="${PYTHON:-/home/yl2428/.conda/envs/video-llama/bin/python}"
CONFIG_TEMPLATE="${REPO}/configs/eval_2_1/pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml"

mkdir -p "$OUT_DIR"
MERGED="${OUT_DIR}/merged_for_cache.pt"

if [[ ! -f "$MERGED" ]]; then
  echo "[1/3] merging LoRA weights from $LORA_CKPT"
  cd "$REPO"
  "$PYTHON" scripts/pd_hand/merge_lora_checkpoint.py --in "$LORA_CKPT" --out "$MERGED"
else
  echo "[1/3] merged checkpoint exists: $MERGED"
fi

# Write a temp config that points at the merged checkpoint
TMPCFG="${OUT_DIR}/cache_config_lora.yaml"
"$PYTHON" - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("${CONFIG_TEMPLATE}").read())
cfg["model_kwargs"]["checkpoint"] = "${MERGED}"
Path("${TMPCFG}").write_text(yaml.dump(cfg, sort_keys=False))
print("wrote", "${TMPCFG}")
PY

echo "[2/3] launching cache jobs (4 GPUs in parallel)"
cd "$REPO"
for SETUP in nocrop_mean handcrop_mean; do
  case "$SETUP" in
    nocrop_mean)   POOL=mean;     CROP="";          GPU=0 ;;
    handcrop_mean) POOL=mean;     CROP="--hand-crop"; GPU=1 ;;
  esac
  LOG="${OUT_DIR}/cache_${SETUP}.log"
  echo "  $SETUP -> cuda:$GPU log=$LOG"
  CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
    --config "$TMPCFG" \
    --out-dir "$OUT_DIR" \
    --split both --batch-size 2 --num-workers 4 \
    --pool "$POOL" $CROP --device cuda:0 \
    > "$LOG" 2>&1 &
done
wait
echo "[3/3] cache done"
ls -la "$OUT_DIR"/*.npz

# Build the handcrop+nocrop concat .npz (same as the original pipeline)
"$PYTHON" - <<PY
import numpy as np
from pathlib import Path
ROOT = Path("${OUT_DIR}")
for split in ["train", "val"]:
    hc_files = sorted(ROOT.glob(f"vjepa21_vitl384_{split}_*_handcrop_mean.npz"))
    nc_files = sorted(ROOT.glob(f"vjepa21_vitl384_{split}_*_nocrop_mean.npz"))
    if not hc_files or not nc_files:
        print(f"{split}: missing inputs"); continue
    hc = np.load(hc_files[0]); nc = np.load(nc_files[0])
    out_name = hc_files[0].name.replace("_handcrop_mean", "_handcrop_nocrop_mean_concat")
    out_path = ROOT / out_name
    np.savez_compressed(out_path,
        x=np.concatenate([hc["x"].astype(np.float16), nc["x"].astype(np.float16)], axis=-1),
        y=hc["y"])
    print(f"wrote {out_path} x.shape={hc['x'].shape[:2] + (2 * hc['x'].shape[-1],)}")
PY
echo "concat done"
