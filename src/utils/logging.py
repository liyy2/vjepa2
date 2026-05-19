# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys
from collections.abc import Mapping

import torch


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        # -- print headers
        with open(self.fname, mode) as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class WandBLogger(object):
    """Small wandb wrapper with the same field-spec/log shape as CSVLogger."""

    def __init__(self, *argv, **kwargs):
        self.keys = [v[1] for v in argv]
        self.enabled = kwargs.pop("enabled", True)
        self.step_key = kwargs.pop("step_key", self.keys[0] if self.keys else None)
        self._wandb = None
        self.run = None
        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb is not installed; disable wandb logging or install wandb.") from exc

        self._wandb = wandb
        init_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self.run = wandb.init(**init_kwargs)
        if self.step_key is not None:
            wandb.define_metric(self.step_key)
            wandb.define_metric("*", step_metric=self.step_key)

    def log(self, *argv):
        if not self.enabled or self.run is None:
            return
        payload = {
            key: _to_wandb_value(value)
            for key, value in zip(self.keys, argv)
        }
        self._wandb.log(payload)

    def finish(self):
        if self.enabled and self.run is not None:
            self._wandb.finish()
            self.run = None


def _to_wandb_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {k: _to_wandb_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wandb_value(v) for v in value]
    return value


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
