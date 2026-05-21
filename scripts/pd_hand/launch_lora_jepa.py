#!/usr/bin/env python
"""Direct launcher for V-JEPA 2.1 LoRA adaptation via torchrun (no submitit).

Run with:
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29551 \\
      scripts/pd_hand/launch_lora_jepa.py \\
      configs/train_2_1/pd_hand/vitl384-lora-jepa-adapt-32f-step1-fold0.yaml

The training app's `init_distributed` only checks for SLURM_* env vars. When
launched under torchrun (which sets RANK / WORLD_SIZE / LOCAL_RANK), we shim
SLURM-equivalents before calling into the app.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

# Make sure the repo root is on sys.path so `app.scaffold` is importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _shim_slurm_envs():
    """If we're under torchrun, overwrite SLURM_* env vars to match torchrun's RANK.

    Important: when running inside an existing SLURM allocation (e.g. an OOD
    session), SLURM_PROCID / SLURM_LOCALID are already set to 0 for the parent
    job; those values are NOT what we want for the torchrun subprocesses.
    Override unconditionally.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    os.environ["SLURM_JOB_ID"] = os.environ.get("TORCHELASTIC_RUN_ID", os.environ.get("SLURM_JOB_ID", "0"))
    os.environ["SLURM_NTASKS"] = os.environ["WORLD_SIZE"]
    os.environ["SLURM_PROCID"] = os.environ["RANK"]
    os.environ["SLURM_LOCALID"] = os.environ.get("LOCAL_RANK", "0")
    os.environ.setdefault("HOSTNAME", socket.gethostname())
    # The training app hardcodes `cuda:0`; pin each rank to a single GPU so they don't collide.
    if "CUDA_VISIBLE_DEVICES" not in os.environ or "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: launch_lora_jepa.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    _shim_slurm_envs()
    import yaml
    from app.scaffold import main as app_main
    cfg_path = sys.argv[1]
    params = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
    app_main(params["app"], args=params)
