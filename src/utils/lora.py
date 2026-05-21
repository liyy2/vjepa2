# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA adapter around an existing nn.Linear layer.

    The wrapped base layer is kept intact under ``.base`` so checkpoint loading
    can map normal Linear weights to ``base.weight`` / ``base.bias``.
    """

    def __init__(self, base_layer, rank, alpha=None, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(self.rank if alpha is None else alpha)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.register_buffer(
            "lora_scaling",
            torch.tensor(self.alpha / self.rank, dtype=torch.float32),
            persistent=True,
        )
        self.reset_lora_parameters()

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        y = self.base(x)
        z = F.linear(self.dropout(x), self.lora_A)
        z = F.linear(z, self.lora_B) * self.lora_scaling.to(device=z.device, dtype=z.dtype)
        return y + z.to(dtype=y.dtype)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, bias={self.bias is not None}"
        )


def inject_lora(model, target_modules=("attn.qkv", "attn.proj"), rank=8, alpha=None, dropout=0.0):
    """Replace selected Linear modules with LoRALinear.

    A target matches when the full module name is exactly the target, ends with
    the target suffix, or the target is ``*``.
    """

    target_modules = tuple(target_modules or ())
    replaced = []

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full_name = child_name if parent_name == "" else f"{parent_name}.{child_name}"
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            if not _matches_lora_target(full_name, target_modules):
                continue
            setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced.append(full_name)

    return replaced


def mark_only_lora_as_trainable(
    model,
    train_bias="none",
    train_layer_norm=False,
    train_embeddings=False,
    train_patch_embed=False,
    extra_trainable_keywords=None,
):
    """Freeze a model except LoRA params and explicitly requested small params."""

    extra_trainable_keywords = tuple(extra_trainable_keywords or ())
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        trainable = ("lora_A" in name) or ("lora_B" in name)
        if train_bias == "all" and name.endswith(".bias"):
            trainable = True
        if train_layer_norm and _is_layer_norm_param(name):
            trainable = True
        if train_embeddings and _is_embedding_param(name):
            trainable = True
        if train_patch_embed and "patch_embed" in name:
            trainable = True
        if extra_trainable_keywords and any(keyword in name for keyword in extra_trainable_keywords):
            trainable = True
        param.requires_grad = trainable

    return count_trainable_parameters(model)


def count_trainable_parameters(model):
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def has_lora_state_dict(state_dict):
    return any(key.endswith(".lora_A") or key.endswith(".lora_B") for key in state_dict)


def remap_state_dict_for_lora(model_state_dict, pretrained_state_dict):
    """Map plain Linear checkpoint keys to LoRALinear ``.base`` keys when needed."""

    remapped = dict(pretrained_state_dict)
    for model_key, model_value in model_state_dict.items():
        if model_key in remapped:
            continue
        for candidate in _checkpoint_key_candidates(model_key):
            value = pretrained_state_dict.get(candidate)
            if value is not None and tuple(value.shape) == tuple(model_value.shape):
                remapped[model_key] = value
                break
    return remapped


def merge_lora_state_dict(state_dict):
    """Merge LoRA weights into normal Linear weights in a state dict.

    This allows a LoRA-adapted checkpoint to be loaded by the original V-JEPA
    encoder without constructing LoRA modules at evaluation time.
    """

    if not has_lora_state_dict(state_dict):
        return state_dict

    merged = OrderedDict()
    consumed = set()

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if key.endswith(".base.weight"):
            prefix = key[: -len("base.weight")]
            out_key = f"{prefix}weight"
            lora_A = state_dict.get(f"{prefix}lora_A")
            lora_B = state_dict.get(f"{prefix}lora_B")
            scaling = state_dict.get(f"{prefix}lora_scaling")
            if lora_A is not None and lora_B is not None:
                scale = float(scaling.item()) if torch.is_tensor(scaling) else 1.0
                delta = torch.matmul(lora_B.float(), lora_A.float()) * scale
                value = value + delta.to(device=value.device, dtype=value.dtype)
                consumed.update({f"{prefix}lora_A", f"{prefix}lora_B", f"{prefix}lora_scaling"})
            merged[out_key] = value
            consumed.add(key)
        elif key.endswith(".base.bias"):
            prefix = key[: -len("base.bias")]
            merged[f"{prefix}bias"] = value
            consumed.add(key)
        elif key.endswith(".lora_A") or key.endswith(".lora_B") or key.endswith(".lora_scaling"):
            consumed.add(key)
        else:
            merged[key] = value

    return merged


def _matches_lora_target(name, target_modules):
    for target in target_modules:
        if target == "*" or name == target or name.endswith(target):
            return True
    return False


def _checkpoint_key_candidates(model_key):
    keys = [model_key]
    if ".base." in model_key:
        keys.append(model_key.replace(".base.", "."))

    expanded = []
    for key in keys:
        expanded.append(key)
        if key.startswith("module."):
            expanded.append(key[len("module.") :])
        else:
            expanded.append(f"module.{key}")
    return expanded


def _is_layer_norm_param(name):
    parts = name.split(".")
    return any(part.startswith("norm") for part in parts)


def _is_embedding_param(name):
    return (
        "pos_embed" in name
        or "mod_embed" in name
        or "mask_token" in name
        or "register" in name
    )
