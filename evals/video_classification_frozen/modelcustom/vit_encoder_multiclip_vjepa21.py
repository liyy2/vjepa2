"""
V-JEPA 2.1 multiclip frozen encoder wrapper.

This mirrors ``vit_encoder_multiclip`` but imports the V-JEPA 2.1 encoder
implementation and loads its checkpoint strictly by default.
"""

import logging

import torch
import torch.nn as nn

import app.vjepa_2_1.models.vision_transformer as vit
from app.vjepa_2_1.models.utils.pos_embs import get_1d_sincos_pos_embed
from src.masks.utils import apply_masks
from src.utils.lora import merge_lora_state_dict, inject_lora

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    resolution: int,
    frames_per_clip: int,
    checkpoint: str,
    # --
    model_kwargs: dict,
    wrapper_kwargs: dict,
):
    logger.info(f"Loading V-JEPA 2.1 pretrained model from {checkpoint}")
    checkpoint_data = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = dict(model_kwargs["encoder"])
    enc_ckp_key = enc_kwargs.pop("checkpoint_key", "ema_encoder")
    enc_model_name = enc_kwargs.pop("model_name")
    strict_load = bool(enc_kwargs.pop("strict_state_dict", True))
    # Optional fresh-LoRA injection for supervised fine-tuning. Set
    # lora.enabled=true in model_kwargs.encoder to add LoRA adapters after
    # loading the base checkpoint. The downstream fine-tune patterns
    # ('lora_A', 'lora_B') will then have parameters to unfreeze.
    lora_cfg = enc_kwargs.pop("lora", None)

    model = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        **enc_kwargs,
    )

    # Try the requested key, fall back to common aliases. JEPA-style training
    # saves the EMA weights under "target_encoder"; eval defaults to
    # "ema_encoder". Accept either.
    fallback_keys = [enc_ckp_key, "ema_encoder", "target_encoder", "encoder"]
    chosen_key = next((k for k in fallback_keys if k in checkpoint_data), None)
    if chosen_key is None:
        raise KeyError(
            f"None of {fallback_keys} found in checkpoint; available keys: {list(checkpoint_data)}"
        )
    if chosen_key != enc_ckp_key:
        logger.info(
            f"checkpoint_key '{enc_ckp_key}' not present; using '{chosen_key}' instead"
        )
    pretrained_dict = checkpoint_data[chosen_key]
    pretrained_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in pretrained_dict.items()
    }
    pretrained_dict = merge_lora_state_dict(pretrained_dict)
    msg = model.load_state_dict(pretrained_dict, strict=strict_load)
    logger.info(f"loaded V-JEPA 2.1 encoder with msg: {msg}")

    if lora_cfg and lora_cfg.get("enabled", False):
        target_modules = lora_cfg.get("target_modules", ["attn.qkv", "attn.proj"])
        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 16))
        dropout = float(lora_cfg.get("dropout", 0.05))
        n_inj = inject_lora(
            model,
            target_modules=tuple(target_modules),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        logger.info(
            f"injected LoRA: rank={rank} alpha={alpha} dropout={dropout} target={target_modules} -> {n_inj} modules"
        )
    print(model)

    wrapper_kwargs = dict(wrapper_kwargs)
    return_hierarchical = bool(wrapper_kwargs.pop("return_hierarchical", False))
    output_dim = model.embed_dim
    if return_hierarchical:
        model.return_hierarchical = True
        output_dim = model.embed_dim * len(model.out_layers_distillation)
        logger.info(
            "using V-JEPA 2.1 hierarchical output layers %s; output dim %s",
            model.out_layers_distillation,
            output_dim,
        )

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
        output_dim=output_dim,
        **wrapper_kwargs,
    )
    del checkpoint_data
    return model


class ClipAggregation(nn.Module):
    """Process each clip independently and concatenate all tokens."""

    def __init__(
        self,
        model,
        tubelet_size=2,
        output_dim=None,
        max_frames=128,
        use_pos_embed=False,
        clip_batch_size=None,
        pool_tokens=None,
        return_segment_outputs=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.token_embed_dim = embed_dim = int(output_dim or model.embed_dim)
        self.num_heads = model.num_heads
        self._warned_dynamic_pos_embed = False
        self.clip_batch_size = int(clip_batch_size) if clip_batch_size else None
        if isinstance(pool_tokens, bool):
            pool_tokens = "mean" if pool_tokens else None
        self.pool_tokens = pool_tokens
        self.return_segment_outputs = bool(return_segment_outputs)
        self.embed_dim = self._pooled_embed_dim(embed_dim)

        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            if max_T < 1:
                raise ValueError(f"max_frames={max_frames} is too short for tubelet_size={tubelet_size}")
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _pooled_embed_dim(self, embed_dim):
        if self.pool_tokens in {"mean", "avg", "average"}:
            return embed_dim
        if self.pool_tokens in {"mean_std", "mean+std"}:
            return 2 * embed_dim
        if self.pool_tokens in {"mean_std_max", "mean+std+max"}:
            return 3 * embed_dim
        return embed_dim

    def _pool_tokens(self, x):
        if self.pool_tokens in {"mean", "avg", "average"}:
            return x.mean(dim=1, keepdim=True)
        if self.pool_tokens in {"mean_std", "mean+std"}:
            mean = x.mean(dim=1)
            std = x.float().std(dim=1, unbiased=False).to(dtype=x.dtype)
            return torch.cat([mean, std], dim=1).unsqueeze(1)
        if self.pool_tokens in {"mean_std_max", "mean+std+max"}:
            mean = x.mean(dim=1)
            std = x.float().std(dim=1, unbiased=False).to(dtype=x.dtype)
            max_values = x.max(dim=1).values
            return torch.cat([mean, std, max_values], dim=1).unsqueeze(1)
        return x

    def _pos_embed_for_length(self, length, device, dtype):
        if self.pos_embed is not None and length <= self.pos_embed.shape[1]:
            return self.pos_embed.to(device=device, dtype=dtype)

        if not self._warned_dynamic_pos_embed:
            logger.warning(
                "Extending fixed temporal pos_embed on the fly to length %s; "
                "increase wrapper_kwargs.max_frames to avoid regenerating it.",
                length,
            )
            self._warned_dynamic_pos_embed = True

        sincos = get_1d_sincos_pos_embed(self.embed_dim, length)
        return torch.from_numpy(sincos).float().unsqueeze(0).to(device=device, dtype=dtype)

    def forward(self, x, clip_indices=None):
        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, F, H, W = x[0][0].size()

        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)

        if self.clip_batch_size and x.size(0) > self.clip_batch_size:
            outputs = torch.cat(
                [
                    self.model(xi)
                    for xi in torch.split(x, self.clip_batch_size, dim=0)
                ],
                dim=0,
            )
        else:
            outputs = self.model(x)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size
            S = N // T

            eff_B = B * num_views_per_clip
            if self.return_segment_outputs:
                segment_outputs = []
                for i in range(num_clips):
                    o = outputs[i * eff_B : (i + 1) * eff_B]
                    for j in range(num_views_per_clip):
                        outputs_ij = o[j * B : (j + 1) * B]
                        if (self.pos_embed is not None) and (clip_indices is not None):
                            raw_indices = clip_indices[i][:, :: self.tubelet_size]
                            temporal_indices = (raw_indices // self.tubelet_size).long()
                            max_index = int(temporal_indices.max().item())
                            pos_embed = self._pos_embed_for_length(
                                max_index + 1,
                                device=outputs_ij.device,
                                dtype=outputs_ij.dtype,
                            ).repeat(B, 1, 1)
                            pos_embed = apply_masks(pos_embed, [temporal_indices], concat=False)[0]
                            pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)
                            outputs_ij = outputs_ij + pos_embed.flatten(1, 2)
                        outputs_ij = self._pool_tokens(outputs_ij)
                        segment_outputs.append(outputs_ij)
                return segment_outputs

            all_outputs = [[] for _ in range(num_views_per_clip)]
            for i in range(num_clips):
                o = outputs[i * eff_B : (i + 1) * eff_B]
                for j in range(num_views_per_clip):
                    all_outputs[j].append(o[j * B : (j + 1) * B])

            for i, outputs_i in enumerate(all_outputs):
                outputs_i = [o.reshape(B, T, S, D) for o in outputs_i]
                outputs_i = torch.cat(outputs_i, dim=1).flatten(1, 2)
                if (self.pos_embed is not None) and (clip_indices is not None):
                    raw_indices = [c[:, :: self.tubelet_size] for c in clip_indices]
                    temporal_indices = [
                        (indices // self.tubelet_size).long()
                        for indices in raw_indices
                    ]
                    max_index = max(int(indices.max().item()) for indices in temporal_indices)
                    pos_embed = self._pos_embed_for_length(
                        max_index + 1,
                        device=outputs_i.device,
                        dtype=outputs_i.dtype,
                    ).repeat(B, 1, 1)
                    pos_embed = apply_masks(pos_embed, temporal_indices, concat=False)
                    pos_embed = torch.cat(pos_embed, dim=1)
                    pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, S, 1)
                    pos_embed = pos_embed.flatten(1, 2)
                    outputs_i += pos_embed
                outputs_i = self._pool_tokens(outputs_i)
                all_outputs[i] = outputs_i

            return all_outputs

        return multiviews_postprocess(outputs)
