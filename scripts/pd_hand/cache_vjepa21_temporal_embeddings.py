#!/usr/bin/env python
"""Cache pure-vision V-JEPA 2.1 temporal embeddings for PD hand clips.

The cached tensor is a per-sample sequence of spatially pooled V-JEPA tubelet
features.  It intentionally does not include side, dx, item 3.5 labels, or
MediaPipe kinematic measurements.
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
    "pd_hand_item_3_4_fold0_vjepa2_1_vitl384_corn_accuracy_dense.yaml"
)
DEFAULT_OUT = (
    "/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/"
    "foundation_model_minimal_hand_tasks/vjepa2_embeddings/item_3_4/fold_0"
)
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hand-crop", action="store_true", help="Use ClipDataset visual hand crop preprocessing.")
    parser.add_argument("--pool", choices=("mean", "mean_std"), default="mean")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg["experiment"]
    data_cfg = dict(exp["data"])
    model_cfg = cfg["model_kwargs"]
    frames_per_clip = int(data_cfg["frames_per_clip"])
    frame_step = int(data_cfg["frame_step"])
    num_segments = int(data_cfg["num_segments"])
    resolution = int(data_cfg["resolution"])
    patch_size = int(model_cfg["pretrain_kwargs"]["encoder"].get("patch_size", 16))
    spatial_tokens = (resolution // patch_size) ** 2

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

    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", data_cfg["dataset_train"]))
    if args.split in ("val", "both"):
        splits.append(("val", data_cfg["dataset_val"]))

    for split_name, csv_path in splits:
        suffix = (
            f"vjepa21_vitl384_{split_name}_"
            f"{frames_per_clip}f_step{frame_step}_{num_segments}seg_"
            f"{'handcrop' if args.hand_crop else 'nocrop'}_{args.pool}.npz"
        )
        out_path = out_dir / suffix
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
            pool=args.pool,
            device=device,
            config={"config": args.config, "split": split_name, "data": data_cfg, "pool": args.pool},
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
    pool: str,
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
    )

    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
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
            seq = spatial_pool(tokens, spatial_tokens=spatial_tokens, pool=pool)
            embeddings.append(seq.cpu().numpy().astype(np.float16))
            labels.append(data[1].cpu().numpy().astype(np.int64))
            coverage.extend(_batch_coverage_rows(data[3]))
            if batch_idx % 10 == 0 or batch_idx == len(loader):
                print(f"{out_path.name}: batch {batch_idx}/{len(loader)}", flush=True)

    x = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        coverage=np.asarray([json.dumps(row) for row in coverage]),
        config=json.dumps(config),
        temporal_length=np.asarray([x.shape[1]], dtype=np.int64),
        input_dim=np.asarray([x.shape[2]], dtype=np.int64),
    )
    print(f"wrote {out_path} x={x.shape} y={y.shape}")


def spatial_pool(tokens: torch.Tensor, spatial_tokens: int, pool: str) -> torch.Tensor:
    batch_size, num_tokens, embed_dim = tokens.shape
    if num_tokens % spatial_tokens != 0:
        raise ValueError(f"num_tokens={num_tokens} is not divisible by spatial_tokens={spatial_tokens}")
    temporal_tokens = num_tokens // spatial_tokens
    x = tokens.reshape(batch_size, temporal_tokens, spatial_tokens, embed_dim)
    mean = x.mean(dim=2)
    if pool == "mean":
        return mean
    std = x.std(dim=2, unbiased=False)
    return torch.cat([mean, std], dim=-1)


def _batch_coverage_rows(batch_coverage: Any) -> list[dict[str, float]]:
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


if __name__ == "__main__":
    main()
