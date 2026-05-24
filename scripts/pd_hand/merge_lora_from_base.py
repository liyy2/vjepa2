#!/usr/bin/env python
"""Merge a partial-state LoRA-FT checkpoint into a base V-JEPA checkpoint.

The supervised LoRA-FT pipeline only saves the trainable LoRA params
(lora_A, lora_B) under the `encoder` key, not the full encoder state.
The standard `merge_lora_checkpoint.py` expects a full state dict
(<prefix>.base.weight + <prefix>.lora_A/B). This script instead:

1. Loads the BASE V-JEPA checkpoint (full encoder weights),
2. Loads the LoRA delta from the LoRA-FT checkpoint,
3. Computes per-module delta = lora_B @ lora_A * scaling,
4. Adds the delta into the base weight,
5. Saves a fully-mergeable checkpoint with `ema_encoder` and `encoder`
   keys, suitable for the eval pipeline.

The base ema_encoder keys are prefixed `module.backbone.`; the LoRA keys
are prefixed `model.`. We strip both and match by the shared module path
(e.g., `blocks.0.attn.qkv`).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def _strip_prefix(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base V-JEPA checkpoint (has ema_encoder/encoder keys)")
    p.add_argument("--lora", required=True, help="LoRA-FT checkpoint (saved trainable params only)")
    p.add_argument("--out", required=True, help="Output merged checkpoint")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-rank", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    lora = torch.load(args.lora, map_location="cpu", weights_only=False)

    # Pull ema_encoder from base; strip "module.backbone."
    base_sd_raw = base.get("ema_encoder") or base.get("encoder")
    if base_sd_raw is None:
        raise SystemExit("base checkpoint has no ema_encoder/encoder key")
    base_sd = _strip_prefix(base_sd_raw, "module.backbone.")
    base_sd = _strip_prefix(base_sd, "module.")
    base_sd = _strip_prefix(base_sd, "backbone.")

    # Pull encoder from LoRA-FT; strip "module." then "model."
    lora_sd_raw = lora.get("encoder")
    if lora_sd_raw is None:
        raise SystemExit("lora checkpoint has no 'encoder' key")
    lora_sd = _strip_prefix(lora_sd_raw, "module.")
    lora_sd = _strip_prefix(lora_sd, "model.")

    # Compute scaling = alpha / rank
    scale = float(args.lora_alpha) / float(args.lora_rank)

    merged = dict(base_sd)
    applied = 0
    skipped = 0
    for key, A in lora_sd.items():
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len(".lora_A")]
        B_key = f"{prefix}.lora_B"
        B = lora_sd.get(B_key)
        if B is None:
            skipped += 1
            continue
        # Target base weight: prefix + ".weight"
        target_key = f"{prefix}.weight"
        if target_key not in merged:
            skipped += 1
            print(f"  [warn] base key missing for LoRA delta: {target_key}")
            continue
        # delta = B @ A * scale  (shape: [out, in] = [out, r] @ [r, in])
        W = merged[target_key]
        delta = (B.float() @ A.float()) * scale
        if tuple(delta.shape) != tuple(W.shape):
            print(f"  [warn] shape mismatch for {target_key}: W={tuple(W.shape)} delta={tuple(delta.shape)}; skipping")
            skipped += 1
            continue
        merged[target_key] = W + delta.to(dtype=W.dtype)
        applied += 1

    print(f"merged {applied} LoRA deltas; skipped {skipped}")
    print(f"final merged state dict has {len(merged)} tensors")

    out = {"ema_encoder": merged, "encoder": merged}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"wrote {out_path}")
    import os
    print(f"size: {os.path.getsize(out_path)/(1024**2):.1f} MB")


if __name__ == "__main__":
    main()
