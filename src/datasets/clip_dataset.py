"""Manifest-backed hand-clip dataset for downstream V-JEPA probing."""

from __future__ import annotations

import math
import os
import pathlib
import warnings
from logging import getLogger
from typing import Any

import numpy as np
import pandas as pd
import torch
from decord import cpu, VideoReader

from src.datasets.utils.dataloader import (
    ConcatIndices,
    MonitoredDataset,
    NondeterministicDataLoader,
)
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

logger = getLogger()

SKIP_FLAG_TOKENS = ("bad", "exclude", "invalid", "missing", "qc_fail")


def make_clipdataset(
    data_paths,
    batch_size,
    frames_per_clip=16,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    hand_crop=True,
    crop_size=256,
    crop_scale=2.35,
    metadata_columns=None,
    metadata_categories=None,
    adaptive_num_clips=False,
    adaptive_duration_col="duration_s",
    adaptive_max_clips=None,
    adaptive_min_clips=1,
):
    if adaptive_num_clips and collator is None:
        collator = adaptive_clip_collate
    dataset = ClipDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        hand_crop=hand_crop,
        crop_size=crop_size,
        crop_scale=crop_scale,
        metadata_columns=metadata_columns,
        metadata_categories=metadata_categories,
        adaptive_num_clips=adaptive_num_clips,
        adaptive_duration_col=adaptive_duration_col,
        adaptive_max_clips=adaptive_max_clips,
        adaptive_min_clips=adaptive_min_clips,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    loader_class = torch.utils.data.DataLoader if deterministic else NondeterministicDataLoader
    data_loader = loader_class(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    logger.info("ClipDataset data loader created")
    return dataset, data_loader, dist_sampler


class ClipDataset(torch.utils.data.Dataset):
    """Video classification dataset using the PD hand-task manifest schema."""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,
        label_col="label",
        clip_col="clip_path",
        side_col="side",
        flags_col="flags",
        hand_crop=True,
        crop_size=256,
        crop_scale=2.35,
        metadata_columns=None,
        metadata_categories=None,
        adaptive_num_clips=False,
        adaptive_duration_col="duration_s",
        adaptive_max_clips=None,
        adaptive_min_clips=1,
    ):
        self.data_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.label_col = label_col
        self.clip_col = clip_col
        self.side_col = side_col
        self.flags_col = flags_col
        self.hand_crop = hand_crop
        self.crop_size = int(crop_size)
        self.crop_scale = float(crop_scale)
        self.metadata_columns = list(metadata_columns or [])
        self.metadata_categories = metadata_categories or {}
        self.adaptive_num_clips = bool(adaptive_num_clips)
        self.adaptive_duration_col = adaptive_duration_col
        self.adaptive_max_clips = adaptive_max_clips
        self.adaptive_min_clips = max(1, int(adaptive_min_clips or 1))

        if sum(v is not None for v in (fps, duration, frame_step)) != 1:
            raise ValueError(
                f"Must specify exactly one of either {fps=}, {duration=}, or {frame_step=}."
            )

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in self.data_paths]
        else:
            if len(dataset_fpcs) != len(self.data_paths):
                raise ValueError("Frames per clip must match the number of data paths.")
            self.dataset_fpcs = dataset_fpcs

        samples: list[str] = []
        labels: list[int] = []
        metadata: list[dict[str, Any]] = []
        self.num_samples_per_dataset: list[int] = []
        for data_path in self.data_paths:
            frame = pd.read_csv(data_path)
            missing = {clip_col, label_col} - set(frame.columns)
            if missing:
                raise ValueError(f"{data_path} is missing required columns: {sorted(missing)}")
            frame = frame.copy()
            frame[clip_col] = frame[clip_col].astype(str)
            frame[label_col] = pd.to_numeric(frame[label_col], errors="coerce")
            frame = frame[frame[label_col].notna()]
            frame = frame[frame[clip_col].map(os.path.exists)]
            if flags_col in frame.columns:
                keep = ~frame[flags_col].fillna("").astype(str).str.lower().map(_has_skip_flag)
                frame = frame[keep]
            samples.extend(frame[clip_col].tolist())
            labels.extend(frame[label_col].astype(int).tolist())
            metadata.extend(frame.to_dict("records"))
            self.num_samples_per_dataset.append(len(frame))

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)
        self.sample_weights = None
        if datasets_weights is not None:
            self.sample_weights = []
            for dataset_weight, num_samples in zip(datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dataset_weight / max(1, num_samples)] * num_samples

        self.samples = samples
        self.labels = labels
        self.metadata = metadata
        if not self.samples:
            raise ValueError("ClipDataset has no usable clips after manifest filtering.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        while not loaded_sample:
            if not isinstance(sample, str):
                logger.warning("Invalid sample.")
            else:
                loaded_sample = self.get_item_video(index)
            if not loaded_sample:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]
        return loaded_sample

    def get_item_video(self, index):
        sample = self.samples[index]
        label = int(self.labels[index])
        metadata = self.metadata[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]

        buffer, clip_indices, coverage = self.loadvideo_decord(sample, frames_per_clip, metadata)
        if len(buffer) == 0:
            return
        coverage["sample_index"] = int(index)

        side = str(metadata.get(self.side_col, "") or "")
        if self.hand_crop:
            buffer = _crop_to_hand(buffer, side=side, crop_size=self.crop_size, crop_scale=self.crop_scale)

        def split_into_clips(video):
            fpc = frames_per_clip
            return [video[i * fpc : (i + 1) * fpc] for i in range(len(clip_indices))]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        if getattr(self, "metadata_columns", []):
            return buffer, label, clip_indices, coverage, self._metadata_features(metadata)
        return buffer, label, clip_indices, coverage

    def loadvideo_decord(self, sample, fpc, metadata=None):
        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f"video path not found {fname=}")
            return [], None, None

        file_size = os.path.getsize(fname)
        if file_size > self.filter_long_videos:
            warnings.warn(f"skipping long video of size {file_size=} bytes")
            return [], None, None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None, None

        frame_step = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as exc:
                logger.warning(exc)
                video_fps = 30

            if self.duration is not None:
                frame_step = int(self.duration * video_fps / fpc)
            else:
                frame_step = video_fps // self.fps

        assert frame_step is not None and frame_step > 0
        clip_len = int(fpc * frame_step)

        if self.adaptive_num_clips:
            return self._loadvideo_decord_adaptive(
                vr=vr,
                fpc=fpc,
                frame_step=frame_step,
                clip_len=clip_len,
                metadata=metadata or {},
            )

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"skipping video of length {len(vr)}")
            return [], None, None

        # Guard pathological short videos. Without this, partition_len can be 0
        # (when len(vr) < num_clips) and np.clip(..., 0, partition_len - 1) maps
        # all indices to -1, leading to junk reads from get_batch.
        if len(vr) < max(frame_step, self.num_clips):
            warnings.warn(
                f"skipping video of length {len(vr)} (frame_step={frame_step}, num_clips={self.num_clips})"
            )
            return [], None, None

        vr.seek(0)
        partition_len = len(vr) // self.num_clips
        all_indices, clip_indices = [], []
        for clip_idx in range(self.num_clips):
            if partition_len > clip_len:
                if self.random_clip_sampling:
                    end_index = clip_len
                    end_index = np.random.randint(clip_len, partition_len)
                    start_index = end_index - clip_len
                    start_index = start_index + clip_idx * partition_len
                    end_index = end_index + clip_idx * partition_len
                else:
                    if self.num_clips > 1 and len(vr) > clip_len:
                        start_index = int(round(clip_idx * (len(vr) - clip_len) / (self.num_clips - 1)))
                    else:
                        start_index = 0
                    end_index = start_index + clip_len
                indices = np.linspace(start_index, end_index, num=fpc)
                indices = np.clip(indices, start_index, end_index - 1).astype(np.int64)
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // frame_step)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // frame_step) * partition_len))
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + clip_idx * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // frame_step)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // frame_step) * sample_len))
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len and self.num_clips > 1:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + clip_idx * clip_step
            clip_indices.append(indices)
            all_indices.extend(list(indices))

        coverage = _frame_coverage(all_indices, total_frames=len(vr))
        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices, coverage

    def _loadvideo_decord_adaptive(self, vr, fpc, frame_step, clip_len, metadata):
        total_frames = len(vr)
        if total_frames < max(frame_step, 1):
            warnings.warn(f"skipping video of length {total_frames} (frame_step={frame_step})")
            return [], None, None

        duration_s = _metadata_duration_seconds(metadata, self.adaptive_duration_col)
        if duration_s is None:
            try:
                video_fps = float(vr.get_avg_fps() or 30.0)
            except Exception:
                video_fps = 30.0
            duration_s = total_frames / max(video_fps, 1.0e-6)
        requested_num_clips = max(self.adaptive_min_clips, int(math.ceil(float(duration_s))))
        num_clips = requested_num_clips
        if self.adaptive_max_clips is not None:
            num_clips = min(num_clips, int(self.adaptive_max_clips))

        max_start = max(0, total_frames - clip_len)
        # Always linspace so:
        # (a) when num_clips * clip_len <= total_frames we get even non-overlapping spacing,
        # (b) when num_clips * clip_len >  total_frames adjacent segments slightly overlap
        #     uniformly (avoids the previous bug where the final K starts all collapsed to
        #     max_start, producing duplicate segments).
        if num_clips > 1:
            starts = np.linspace(0, max_start, num=num_clips).round().astype(np.int64).tolist()
        else:
            starts = [0]
        all_indices, clip_indices = [], []
        for start_index in starts:
            indices = start_index + np.arange(fpc, dtype=np.int64) * int(frame_step)
            indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
            clip_indices.append(indices)
            all_indices.extend(indices.tolist())

        coverage = _frame_coverage(all_indices, total_frames=total_frames)
        coverage["num_segments"] = int(num_clips)
        coverage["adaptive_requested_segments"] = int(requested_num_clips)
        coverage["adaptive_duration_s"] = float(duration_s)
        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices, coverage

    def _metadata_features(self, metadata: dict[str, Any]) -> torch.Tensor:
        features: list[float] = []
        for column in getattr(self, "metadata_columns", []):
            value = metadata.get(column, "")
            categories = getattr(self, "metadata_categories", {}).get(column)
            if categories is not None:
                value = str(value)
                features.extend([1.0 if value == str(category) else 0.0 for category in categories])
            else:
                try:
                    features.append(float(value))
                except (TypeError, ValueError):
                    features.append(0.0)
        return torch.tensor(features, dtype=torch.float32)


def _has_skip_flag(flags: str) -> bool:
    text = str(flags).strip().lower()
    if not text:
        return False
    return any(token in text for token in SKIP_FLAG_TOKENS)


def _metadata_duration_seconds(metadata: dict[str, Any], duration_col: str | None) -> float | None:
    candidates = []
    if duration_col:
        candidates.append(duration_col)
    candidates.extend(["duration_s", "clip_duration_s", "duration", "end_s"])
    for key in candidates:
        if key not in metadata:
            continue
        try:
            value = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            if key == "end_s" and "start_s" in metadata:
                try:
                    start_s = float(metadata.get("start_s"))
                    if math.isfinite(start_s):
                        value = value - start_s
                except (TypeError, ValueError):
                    pass
            if value > 0:
                return value
    return None


def adaptive_clip_collate(batch):
    """Pad variable-length per-video clip windows and emit a segment mask.

    Returns the same leading tuple as the fixed-grid ClipDataset:
    ``clips, labels, clip_indices, coverage``. A boolean ``segment_mask`` is
    appended as the last tensor; metadata features, when present, remain before
    the mask.
    """
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("adaptive_clip_collate received an empty batch")

    labels = torch.tensor([int(sample[1]) for sample in batch], dtype=torch.long)
    max_segments = max(len(sample[0]) for sample in batch)
    num_views = len(batch[0][0][0])
    segment_mask = torch.zeros(len(batch), max_segments, dtype=torch.bool)

    clips = []
    clip_indices = []
    for segment_idx in range(max_segments):
        views = []
        for view_idx in range(num_views):
            tensors = []
            for batch_idx, sample in enumerate(batch):
                sample_clips = sample[0]
                if segment_idx < len(sample_clips):
                    tensors.append(sample_clips[segment_idx][view_idx])
                    segment_mask[batch_idx, segment_idx] = True
                else:
                    tensors.append(torch.zeros_like(sample_clips[0][view_idx]))
            views.append(torch.stack(tensors, dim=0))
        clips.append(views)

        index_tensors = []
        for sample in batch:
            sample_indices = sample[2]
            if segment_idx < len(sample_indices):
                index_tensors.append(torch.as_tensor(sample_indices[segment_idx], dtype=torch.long))
            else:
                index_tensors.append(torch.zeros_like(torch.as_tensor(sample_indices[0], dtype=torch.long)))
        clip_indices.append(torch.stack(index_tensors, dim=0))

    coverage = _collate_coverage([sample[3] for sample in batch])
    has_metadata = len(batch[0]) > 4
    if has_metadata:
        metadata_features = torch.stack([sample[4] for sample in batch], dim=0)
        return clips, labels, clip_indices, coverage, metadata_features, segment_mask
    return clips, labels, clip_indices, coverage, segment_mask


def _collate_coverage(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[Any]]:
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = [row.get(key, float("nan")) for row in rows]
        try:
            out[key] = torch.tensor(values, dtype=torch.float32)
        except (TypeError, ValueError):
            out[key] = values
    return out


def _frame_coverage(indices: list[int], total_frames: int) -> dict[str, int | float]:
    unique_indices = np.unique(np.asarray(indices, dtype=np.int64)) if indices else np.asarray([], dtype=np.int64)
    if unique_indices.size == 0 or total_frames <= 0:
        return {
            "video_num_frames": int(max(total_frames, 0)),
            "selected_frame_count": int(len(indices)),
            "unique_frame_count": 0,
            "coverage_rate": 0.0,
            "temporal_span_frame_count": 0,
            "temporal_span_rate": 0.0,
            "first_frame": -1,
            "last_frame": -1,
        }
    first_frame = int(unique_indices.min())
    last_frame = int(unique_indices.max())
    temporal_span = last_frame - first_frame + 1
    return {
        "video_num_frames": int(total_frames),
        "selected_frame_count": int(len(indices)),
        "unique_frame_count": int(unique_indices.size),
        "coverage_rate": float(unique_indices.size / total_frames),
        "temporal_span_frame_count": int(temporal_span),
        "temporal_span_rate": float(temporal_span / total_frames),
        "first_frame": first_frame,
        "last_frame": last_frame,
    }


def _crop_to_hand(buffer: np.ndarray, side: str, crop_size: int, crop_scale: float) -> np.ndarray:
    bbox = _mediapipe_hand_bbox(buffer, side=side, crop_size=crop_size, crop_scale=crop_scale)
    if bbox is None:
        return buffer
    x1, y1, x2, y2 = bbox
    cropped = []
    for frame in buffer:
        cropped.append(_crop_with_padding(frame, x1, y1, x2, y2))
    return np.stack(cropped, axis=0)


def _mediapipe_hand_bbox(
    buffer: np.ndarray,
    side: str,
    crop_size: int,
    crop_scale: float,
) -> tuple[int, int, int, int] | None:
    try:
        import mediapipe as mp
    except Exception:
        return None

    # MediaPipe handedness is from the camera's perspective; the manifest's
    # `side` is from the patient's perspective. They are mirror images, so
    # swap when comparing.
    _SIDE_TO_MEDIAPIPE = {"Left": "Right", "Right": "Left"}
    target_side = _SIDE_TO_MEDIAPIPE.get(side.capitalize(), side.capitalize())
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    wrist_xs: list[float] = []
    wrist_ys: list[float] = []
    radii: list[float] = []
    try:
        stride = max(1, len(buffer) // 12)
        for frame in buffer[::stride]:
            h, w = frame.shape[:2]
            result = hands.process(frame)
            if not result.multi_hand_landmarks:
                continue
            handedness = result.multi_handedness or []
            candidates = []
            for idx, hand_lms in enumerate(result.multi_hand_landmarks):
                label = ""
                if idx < len(handedness):
                    label = handedness[idx].classification[0].label
                points = np.array([(lm.x * w, lm.y * h) for lm in hand_lms.landmark], dtype=np.float32)
                wrist = points[0]
                radius = np.linalg.norm(points - wrist, axis=1).max()
                candidates.append((label, wrist, radius, points))
            if not candidates:
                continue
            chosen = None
            for candidate in candidates:
                if target_side and candidate[0] == target_side:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = candidates[0]
            _, wrist, radius, _ = chosen
            wrist_xs.append(float(wrist[0]))
            wrist_ys.append(float(wrist[1]))
            radii.append(float(radius))
    finally:
        hands.close()

    if not wrist_xs:
        return None
    # Stationary crop: center on the MEDIAN wrist position (robust to outliers),
    # side length = max(crop_size, median_radius * crop_scale * 2).
    # This produces a tight, hand-centered crop that doesn't expand with motion.
    cx = float(np.median(wrist_xs))
    cy = float(np.median(wrist_ys))
    r_med = float(np.median(radii))
    side_len = max(float(crop_size), r_med * crop_scale * 2.0)
    half = side_len / 2.0
    return (
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    )


def _crop_with_padding(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        frame = np.pad(
            frame,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    return frame[y1:y2, x1:x2, :]
