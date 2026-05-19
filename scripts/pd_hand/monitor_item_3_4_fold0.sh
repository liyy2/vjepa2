#!/bin/bash
set -euo pipefail

JOB_ID="${1:-}"
TAG="${TAG:-pd-hand-item-3_4-fold-0}"
RUN_ROOT="${RUN_ROOT:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0_ddp3_h100x3/video_classification_frozen/${TAG}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks/vjepa2_evals/item_3_4/fold_0/run_logs}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"

if [[ -z "$JOB_ID" ]]; then
  JOB_ID="$(squeue -u "${USER}" -h -n vjepa_i34_f0_ddp3 -o "%A" | head -n 1 || true)"
fi

if [[ -n "$JOB_ID" ]]; then
  echo "== squeue =="
  squeue -j "$JOB_ID" -o "%.18i %.8T %.10M %R" || true
  OUT_FILE="${RUN_LOG_DIR}/ddp3_${JOB_ID}.out"
  ERR_FILE="${RUN_LOG_DIR}/ddp3_${JOB_ID}.err"
else
  OUT_FILE="$(ls -t "${RUN_LOG_DIR}"/ddp3_*.out 2>/dev/null | head -n 1 || true)"
  ERR_FILE="${OUT_FILE%.out}.err"
fi

echo "== epoch logs =="
if [[ -f "${OUT_FILE:-}" ]]; then
  tail -120 "$OUT_FILE" | rg '\[INFO|Epoch|run_one_epoch|coverage|CORN|load_checkpoint|wandb' | tail -60 || true
else
  echo "no stdout log found"
fi

echo "== wandb/slurm stderr =="
if [[ -f "${ERR_FILE:-}" ]]; then
  rg -n 'wandb:|ERROR|Traceback|401|offline|Syncing|View run|Resuming run' "$ERR_FILE" | tail -80 || true
else
  echo "no stderr log found"
fi

echo "== csv =="
tail -12 "${RUN_ROOT}/log_r0.csv" 2>/dev/null || true

echo "== metrics_latest =="
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python - <<PY
import json
from pathlib import Path

path = Path("${RUN_ROOT}") / "metrics_latest.json"
if not path.exists():
    print(f"missing {path}")
    raise SystemExit(0)
with path.open() as handle:
    metrics = json.load(handle)
val = metrics.get("val", {})
train = metrics.get("train", {})
print("epoch", metrics.get("epoch"))
print("train_acc", train.get("acc"))
print(
    "val_acc", val.get("acc"),
    "val_accuracy", val.get("accuracy"),
    "spearman", val.get("spearman"),
    "qwk", val.get("quadratic_weighted_kappa"),
    "mae", val.get("mae"),
)
print("confusion", val.get("confusion_matrix"))
print("coverage", {"train": train.get("coverage"), "val": val.get("coverage")})
PY
