#!/bin/bash
set -euo pipefail

REPO="${REPO:-/gpfs/milgram/pi/scherzer/yl2428/vjepa2}"
CONDA="${CONDA:-/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-video-llama}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-${REPO}/wandb_api.txt}"
export HOME="${HOME:-/home/yl2428}"
export WANDB_MODE=online

verify_current_env() {
  "$CONDA" run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import wandb

if not wandb.login(verify=True):
    raise SystemExit(1)
PY
}

if verify_current_env >/tmp/vjepa_wandb_verify.log 2>&1; then
  cat /tmp/vjepa_wandb_verify.log
  echo "wandb_auth=verified_with_${CONDA_ENV}"
  exit 0
fi

if [[ ! -s "$WANDB_API_KEY_FILE" ]]; then
  cat /tmp/vjepa_wandb_verify.log >&2 || true
  echo "wandb_auth=failed missing_or_empty_key_file:${WANDB_API_KEY_FILE}" >&2
  exit 2
fi

tmpdir="$(mktemp -d /tmp/vjepa-wandb-auth.XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

python3 -m venv "$tmpdir/venv"
"$tmpdir/venv/bin/python" -m pip install -q --upgrade pip "wandb>=0.27.0"

WANDB_API_KEY="$(tr -d '\r\n' < "$WANDB_API_KEY_FILE")" \
  "$tmpdir/venv/bin/python" - <<'PY'
import os
import wandb

key = os.environ.get("WANDB_API_KEY", "").strip()
if not key:
    raise SystemExit("WANDB API key file was empty after stripping newlines.")
if not wandb.login(key=key, relogin=True, verify=True):
    raise SystemExit("wandb login verification failed.")
PY

verify_current_env
echo "wandb_auth=bootstrapped_and_verified_with_${CONDA_ENV}"
