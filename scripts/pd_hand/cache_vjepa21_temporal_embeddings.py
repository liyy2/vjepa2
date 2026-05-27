#!/usr/bin/env python
"""Cache V-JEPA 2.1 temporal embeddings for PD hand-task clips.

This is intentionally a thin adapter around Meta's frozen video encoder and the
local ClipDataset.  It writes one NPZ per split with:

  x        [N, T, D] pooled tubelet embeddings
  y        [N] integer labels
  mask     [N, T] valid tubelet mask
  coverage JSON rows from ClipDataset, including sample_index

The cache can be run on a fixed segment grid or on adaptive sliding windows
(`--adaptive-num-clips`) for full-video coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.clip_dataset import make_clipdataset


DEFAULT_CONFIG = (
    "/gpfs/milgram/pi/scherzer/yl2428/vjepa2/configs/eval_2_1/"
    "pd_hand_may24_vitl384_lora_cache.yaml"
)
DEFAULT_OUT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks_may19/vjepa2_embeddings/item_3_4"
)
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--dataset-train", default="", help="Override experiment.data.dataset_train")
    parser.add_argument("--dataset-val", default="", help="Override experiment.data.dataset_val")
    parser.add_argument("--checkpoint", default="", help="Override model_kwargs.checkpoint")
    parser.add_argument("--frames-per-clip", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=0)
    parser.add_argument("--num-segments", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=0)

    parser.add_argument("--hand-crop", action="store_true")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=2.35)
    parser.add_argument(
        "--pool",
        choices=("mean", "mean_std", "max", "mean_max", "topk_mean"),
        default="mean",
        help="Spatial pooling over V-JEPA patch tokens inside each tubelet.",
    )
    parser.add_argument("--topk", type=int, default=16, help="K for topk_mean pooling.")

    parser.add_argument("--adaptive-num-clips", action="store_true")
    parser.add_argument("--adaptive-duration-col", default="duration_s")
    parser.add_argument("--adaptive-min-clips", type=int, default=1)
    parser.add_argument("--adaptive-max-clips", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = dict(cfg["experiment"]["data"])
    model_cfg = dict(cfg["model_kwargs"])

    if args.dataset_train:
        data_cfg["dataset_train"] = args.dataset_train
    if args.dataset_val:
        data_cfg["dataset_val"] = args.dataset_val
    if args.checkpoint:
        model_cfg["checkpoint"] = args.checkpoint
    for attr, key in (
        ("frames_per_clip", "frames_per_clip"),
        ("frame_step", "frame_step"),
        ("num_segments", "num_segments"),
        ("resolution", "resolution"),
    ):
        value = getattr(args, attr)
        if value:
            data_cfg[key] = value

    frames_per_clip = int(data_cfg["frames_per_clip"])
    frame_step = int(data_cfg["frame_step"])
    num_segments = int(data_cfg["num_segments"])
    resolution = int(data_cfg["resolution"])
    patch_size = int(model_cfg["pretrain_kwargs"]["encoder"].get("patch_size", 16))
    spatial_tokens = (resolution // patch_size) ** 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = init_module(
        module_name=model_cfg["module_name"],
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=model_cfg["checkpoint"],
        model_kwargs=model_cfg["pretrain_kwargs"],
        wrapper_kwargs=model_cfg["wrapper_kwargs"],
        device=device,
    )
    encoder.eval()

    split_paths: list[tuple[str, str]] = []
    if args.split in ("train", "both"):
        split_paths.append(("train", data_cfg["dataset_train"]))
    if args.split in ("val", "both"):
        split_paths.append(("val", data_cfg["dataset_val"]))

    for split_name, csv_path in split_paths:
        out_path = out_dir / cache_name(
            split_name=split_name,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            adaptive=args.adaptive_num_clips,
            adaptive_max=args.adaptive_max_clips,
            hand_crop=args.hand_crop,
            crop_size=args.crop_size,
            crop_scale=args.crop_scale,
            pool=args.pool,
        )
        if out_path.exists() and not args.force:
            print(f"exists {out_path}")
            continue
        cache_split(
            encoder=encoder,
            csv_path=csv_path,
            out_path=out_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_segments=num_segments,
            resolution=resolution,
            spatial_tokens=spatial_tokens,
            hand_crop=args.hand_crop,
            crop_size=args.crop_size,
            crop_scale=args.crop_scale,
            pool=args.pool,
            topk=args.topk,
            adaptive_num_clips=args.adaptive_num_clips,
            adaptive_duration_col=args.adaptive_duration_col,
            adaptive_min_clips=args.adaptive_min_clips,
            adaptive_max_clips=args.adaptive_max_clips or None,
            device=device,
            config={
                "config": args.config,
                "split": split_name,
                "data": data_cfg,
                "checkpoint": model_cfg["checkpoint"],
                "pool": args.pool,
                "topk": args.topk,
                "hand_crop": args.hand_crop,
                "crop_size": args.crop_size,
                "crop_scale": args.crop_scale,
                "adaptive_num_clips": args.adaptive_num_clips,
                "adaptive_duration_col": args.adaptive_duration_col,
                "adaptive_min_clips": args.adaptive_min_clips,
                "adaptive_max_clips": args.adaptive_max_clips or None,
            },
        )


def cache_name(
    split_name: str,
    frames_per_clip: int,
    frame_step: int,
    num_segments: int,
    adaptive: bool,
    adaptive_max: int,
    hand_crop: bool,
    crop_size: int,
    crop_scale: float,
    pool: str,
) -> str:
    if adaptive:
        segment_tag = "adaptive"
        if adaptive_max > 0:
            segment_tag += f"_max{adaptive_max}"
    else:
        segment_tag = f"{num_segments}seg"
    crop_tag = ""
    if hand_crop:
        crop_tag = f"_cs{crop_size}_sc{crop_scale:g}"
    view_tag = "handcrop" if hand_crop else "nocrop"
    return (
        f"vjepa21_vitl384_{split_name}_{frames_per_clip}f_step{frame_step}_"
        f"{segment_tag}_{view_tag}{crop_tag}_{pool}.npz"
    )


def cache_split(
    encoder: torch.nn.Module,
    csv_path: str,
    out_path: Path,
    batch_size: int,
    num_workers: int,
    frames_per_clip: int,
    frame_step: int,
    num_segments: int,
    resolution: int,
    spatial_tokens: int,
    hand_crop: bool,
    crop_size: int,
    crop_scale: float,
    pool: str,
    topk: int,
    adaptive_num_clips: bool,
    adaptive_duration_col: str,
    adaptive_min_clips: int,
    adaptive_max_clips: int | None,
    device: torch.device,
    config: dict[str, Any],
) -> None:
    transform = make_transforms(
        training=False,
        num_views_per_clip=1,
        crop_size=resolution,
        normalize=DEFAULT_NORMALIZATION,
    )
    _, loader, _ = make_clipdataset(
        data_paths=[csv_path],
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_segments,
        random_clip_sampling=False,
        allow_clip_overlap=True,
        transform=transform,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        world_size=1,
        rank=0,
        drop_last=False,
        hand_crop=hand_crop,
        crop_size=crop_size,
        crop_scale=crop_scale,
        adaptive_num_clips=adaptive_num_clips,
        adaptive_duration_col=adaptive_duration_col,
        adaptive_min_clips=adaptive_min_clips,
        adaptive_max_clips=adaptive_max_clips,
    )

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    coverage: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader, start=1):
            clips = [
                [view.to(device, non_blocking=True) for view in temporal_clip]
                for temporal_clip in data[0]
            ]
            clip_indices = [indices.to(device, non_blocking=True) for indices in data[2]]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tokens = encoder(clips, clip_indices)[0].float()

            token_mask = token_mask_from_batch(
                data=data,
                tokens=tokens,
                spatial_tokens=spatial_tokens,
                frames_per_clip=frames_per_clip,
                tubelet_size=encoder.tubelet_size,
            )
            seq = spatial_pool(tokens, spatial_tokens=spatial_tokens, pool=pool, topk=topk)
            if token_mask is not None:
                seq = seq.masked_fill(~token_mask.unsqueeze(-1), 0.0)

            seq_np = seq.cpu().numpy().astype(np.float16)
            mask_np = (
                np.ones(seq_np.shape[:2], dtype=bool)
                if token_mask is None
                else token_mask.cpu().numpy().astype(bool)
            )
            embeddings.extend(seq_np)
            masks.extend(mask_np)
            labels.append(data[1].cpu().numpy().astype(np.int64))
            coverage.extend(batch_coverage_rows(data[3]))

            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"{out_path.name}: batch {batch_idx}/{len(loader)}", flush=True)

    x, mask = pad_sequences(embeddings, masks)
    y = np.concatenate(labels, axis=0)
    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        mask=mask,
        coverage=np.asarray([json.dumps(row) for row in coverage]),
        config=json.dumps(config, sort_keys=True),
        temporal_length=np.asarray([x.shape[1]], dtype=np.int64),
        input_dim=np.asarray([x.shape[2]], dtype=np.int64),
    )
    print(f"wrote {out_path} x={x.shape} y={y.shape}")


def spatial_pool(tokens: torch.Tensor, spatial_tokens: int, pool: str, topk: int) -> torch.Tensor:
    batch_size, num_tokens, embed_dim = tokens.shape
    if num_tokens % spatial_tokens != 0:
        raise ValueError(f"num_tokens={num_tokens} is not divisible by spatial_tokens={spatial_tokens}")
    temporal_tokens = num_tokens // spatial_tokens
    x = tokens.reshape(batch_size, temporal_tokens, spatial_tokens, embed_dim)

    if pool == "mean":
        return x.mean(dim=2)
    if pool == "mean_std":
        return torch.cat([x.mean(dim=2), x.std(dim=2, unbiased=False)], dim=-1)
    if pool == "max":
        return x.max(dim=2).values
    if pool == "mean_max":
        return torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=-1)
    if pool == "topk_mean":
        k = min(int(topk), spatial_tokens)
        norms = x.norm(dim=-1)
        _, idx = norms.topk(k=k, dim=2)
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
        return torch.gather(x, dim=2, index=idx).mean(dim=2)
    raise ValueError(f"Unknown spatial pool: {pool}")


def token_mask_from_batch(
    data: Any,
    tokens: torch.Tensor,
    spatial_tokens: int,
    frames_per_clip: int,
    tubelet_size: int,
) -> torch.Tensor | None:
    if len(data) < 5:
        return None
    maybe_mask = data[-1]
    if not (torch.is_tensor(maybe_mask) and maybe_mask.dtype == torch.bool and maybe_mask.ndim == 2):
        return None
    temporal_tokens_per_clip = frames_per_clip // tubelet_size
    token_mask = maybe_mask.repeat_interleave(temporal_tokens_per_clip, dim=1)
    expected_t = tokens.shape[1] // spatial_tokens
    if token_mask.shape[1] != expected_t:
        raise ValueError(f"token_mask length {token_mask.shape[1]} != expected {expected_t}")
    return token_mask.to(device=tokens.device)


def batch_coverage_rows(batch_coverage: Any) -> list[dict[str, float]]:
    if not isinstance(batch_coverage, dict):
        return []
    batch_size = len(next(iter(batch_coverage.values()))) if batch_coverage else 0
    rows = []
    for idx in range(batch_size):
        row = {}
        for key, value in batch_coverage.items():
            try:
                row[key] = float(value[idx])
            except Exception:
                row[key] = float(value)
        rows.append(row)
    return rows


def pad_sequences(sequences: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("No embeddings were produced")
    max_t = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[1]
    x = np.zeros((len(sequences), max_t, dim), dtype=np.float16)
    mask = np.zeros((len(sequences), max_t), dtype=bool)
    for idx, (seq, seq_mask) in enumerate(zip(sequences, masks)):
        t = seq.shape[0]
        x[idx, :t] = seq
        mask[idx, :t] = seq_mask
    return x, mask


if __name__ == "__main__":
    main()
