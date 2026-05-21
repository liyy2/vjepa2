#!/usr/bin/env python
"""Merge LoRA-adapted V-JEPA checkpoint into a base-compatible checkpoint.

The LoRA training script saves `encoder.state_dict()` with the LoRA adapter
parameters intact (lora_A, lora_B, lora_scaling, plus the wrapped base.weight).
For downstream feature extraction with the unmodified encoder, fold the LoRA
deltas (B @ A * scale) into the base weights so the resulting state dict has
the same keys as the original pretrained checkpoint.

Example:
  python scripts/pd_hand/merge_lora_checkpoint.py \\
      --in  <adapt-dir>/latest.pth.tar \\
      --out <adapt-dir>/merged_for_cache.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.utils.lora import merge_lora_state_dict, has_lora_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="LoRA checkpoint (e.g. latest.pth.tar)")
    p.add_argument("--out", required=True, help="Output checkpoint path (base-compatible)")
    p.add_argument("--ema-key", default="target_encoder",
                   help="Key under which the target/EMA encoder lives — used as 'ema_encoder' downstream")
    return p.parse_args()


def _strip_module_prefix(state):
    """Remove 'module.' DDP prefix from keys if present."""
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def _strip_backbone_prefix(state):
    """The training wrappers (MultiSeqWrapper) wrap the encoder; strip leading
    `backbone.` so the result loads cleanly into the bare ViT."""
    cleaned = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            cleaned[k[len("backbone."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def main():
    args = parse_args()
    src = Path(args.in_path)
    if not src.exists():
        sys.exit(f"input checkpoint not found: {src}")
    print(f"loading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    out = {}
    for key in ("encoder", args.ema_key):
        sd = ckpt.get(key)
        if sd is None:
            print(f"  [warn] key '{key}' not in checkpoint; skipping")
            continue
        sd = _strip_module_prefix(sd)
        sd = _strip_backbone_prefix(sd)
        if has_lora_state_dict(sd):
            print(f"  '{key}': merging LoRA deltas into base weights")
            sd = merge_lora_state_dict(sd)
        else:
            print(f"  '{key}': no LoRA keys, copying as-is")
        # The cache script expects `ema_encoder` (per the multiclip_vjepa21 config).
        out_key = "ema_encoder" if key == args.ema_key else key
        out[out_key] = sd

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"wrote {dst}")
    # Summary
    for k, sd in out.items():
        print(f"  {k}: {len(sd)} tensors, first-key={next(iter(sd))}")


if __name__ == "__main__":
    main()
