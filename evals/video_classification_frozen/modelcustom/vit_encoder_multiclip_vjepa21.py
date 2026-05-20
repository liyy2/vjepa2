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

    model = vit.__dict__[enc_model_name](
        img_size=resolution,
        num_frames=frames_per_clip,
        **enc_kwargs,
    )

    pretrained_dict = checkpoint_data[enc_ckp_key]
    pretrained_dict = {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in pretrained_dict.items()
    }
    msg = model.load_state_dict(pretrained_dict, strict=strict_load)
    logger.info(f"loaded V-JEPA 2.1 encoder with msg: {msg}")
    print(model)

    model = ClipAggregation(
        model,
        tubelet_size=model.tubelet_size,
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
        max_frames=128,
        use_pos_embed=False,
    ):
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self._warned_dynamic_pos_embed = False

        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            if max_T < 1:
                raise ValueError(f"max_frames={max_frames} is too short for tubelet_size={tubelet_size}")
            self.pos_embed = nn.Parameter(torch.zeros(1, max_T, embed_dim), requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

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

        outputs = self.model(x)

        def multiviews_postprocess(outputs):
            _, N, D = outputs.size()
            T = F // self.tubelet_size
            S = N // T

            eff_B = B * num_views_per_clip
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
                all_outputs[i] = outputs_i

            return all_outputs

        return multiviews_postprocess(outputs)
