# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import app.vjepa_2_1.models.predictor as vit_pred
import app.vjepa_2_1.models.vision_transformer as video_vit
import torch
import torch.nn.functional as F
import yaml
from app.vjepa_2_1.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.lora import (
    count_trainable_parameters,
    inject_lora,
    mark_only_lora_as_trainable,
    remap_state_dict_for_lora,
)
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat recursively over nested lists."""
    return [
        [[normalize_and_concat(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    import warnings

    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = (
                _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            )
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn(
                    "Configs for online evals must use same number of nodes for slurm-batch processing"
                )

            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu
            _args["pretrain"]["frames_per_clip"] = num_frames
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    pretrained_dict = _select_checkpoint_state(
        checkpoint,
        "encoder",
        prefer_ema=is_anneal,
    )
    if pretrained_dict is not None:
        msg = _load_model_state(encoder, pretrained_dict, "encoder")
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = _select_checkpoint_state(checkpoint, "predictor")
    if pretrained_dict is not None:
        msg = _load_model_state(predictor, pretrained_dict, "predictor")
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")
    else:
        logger.info("checkpoint has no predictor state; leaving predictor initialized")

    if target_encoder is not None:
        pretrained_dict = _select_checkpoint_state(
            checkpoint,
            "target_encoder",
            prefer_ema=is_anneal,
        )
        if pretrained_dict is not None:
            msg = _load_model_state(target_encoder, pretrained_dict, "target_encoder")
            logger.info(
                f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
            )

    if "opt" in checkpoint:
        try:
            opt.load_state_dict(checkpoint["opt"])
        except ValueError:
            print("[warn] Optimizer groups mismatch; reinitializing optimizer.")
    else:
        logger.info("checkpoint has no optimizer state; reinitializing optimizer")
    if scaler is not None:
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        else:
            logger.info("checkpoint has no scaler state; reinitializing scaler")
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def _select_checkpoint_state(checkpoint, role, prefer_ema=False):
    keys_by_role = {
        "encoder": ("encoder", "ema_encoder", "target_encoder"),
        "target_encoder": ("target_encoder", "ema_encoder", "encoder"),
        "predictor": ("predictor",),
    }
    if prefer_ema and role in ("encoder", "target_encoder"):
        keys_by_role = dict(keys_by_role)
        keys_by_role[role] = ("ema_encoder", "target_encoder", "encoder")
    for key in keys_by_role[role]:
        state = checkpoint.get(key)
        if state is not None:
            logger.info(f"loading {role} from checkpoint key '{key}'")
            return state
    return None


def _load_model_state(model, pretrained_dict, role):
    model_state = model.state_dict()
    pretrained_dict = remap_state_dict_for_lora(model_state, pretrained_dict)
    missing_count = 0
    shape_count = 0
    for k, v in model_state.items():
        if k not in pretrained_dict:
            missing_count += 1
            if missing_count <= 20:
                logger.info(f'key "{k}" could not be found in loaded {role} state dict')
        elif pretrained_dict[k].shape != v.shape:
            shape_count += 1
            if shape_count <= 20:
                logger.info(
                    f'key "{k}" is of different shape in model and loaded {role} state dict'
                )
            pretrained_dict[k] = v
    if missing_count > 20:
        logger.info(f"{missing_count - 20} more {role} keys were missing from checkpoint")
    if shape_count > 20:
        logger.info(f"{shape_count - 20} more {role} keys had mismatched shapes")
    return model.load_state_dict(pretrained_dict, strict=False)


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
    lora_kwargs=None,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    lora_kwargs = dict(lora_kwargs or {})
    if bool(lora_kwargs.get("enabled", False)):
        _apply_lora(encoder, predictor, lora_kwargs)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def _apply_lora(encoder, predictor, lora_kwargs):
    rank = int(lora_kwargs.get("rank", 8))
    alpha = lora_kwargs.get("alpha", rank)
    dropout = float(lora_kwargs.get("dropout", 0.0))
    target_modules = lora_kwargs.get("target_modules", ["attn.qkv", "attn.proj"])
    apply_to = lora_kwargs.get("apply_to", ["encoder"])
    if isinstance(apply_to, str):
        apply_to = [apply_to]

    if "encoder" in apply_to or "both" in apply_to:
        replaced = inject_lora(
            encoder,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        logger.info(f"LoRA inserted into {len(replaced)} encoder Linear modules")
        if replaced:
            logger.info(f"first encoder LoRA modules: {replaced[:8]}")
        if bool(lora_kwargs.get("freeze_encoder_base", True)):
            trainable, total = mark_only_lora_as_trainable(
                encoder,
                train_bias=lora_kwargs.get("train_bias", "none"),
                train_layer_norm=bool(lora_kwargs.get("train_layer_norm", False)),
                train_embeddings=bool(lora_kwargs.get("train_embeddings", False)),
                train_patch_embed=bool(lora_kwargs.get("train_patch_embed", False)),
                extra_trainable_keywords=lora_kwargs.get("extra_trainable_keywords", []),
            )
            logger.info(f"Encoder LoRA trainable parameters: {trainable}/{total}")

    if "predictor" in apply_to or "both" in apply_to:
        replaced = inject_lora(
            predictor,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        logger.info(f"LoRA inserted into {len(replaced)} predictor Linear modules")
        if replaced:
            logger.info(f"first predictor LoRA modules: {replaced[:8]}")
        if bool(lora_kwargs.get("freeze_predictor_base", False)):
            trainable, total = mark_only_lora_as_trainable(
                predictor,
                train_bias=lora_kwargs.get("train_bias", "none"),
                train_layer_norm=bool(lora_kwargs.get("train_layer_norm", False)),
                train_embeddings=bool(lora_kwargs.get("train_embeddings", False)),
                train_patch_embed=bool(lora_kwargs.get("train_patch_embed", False)),
                extra_trainable_keywords=lora_kwargs.get("extra_trainable_keywords", []),
            )
            logger.info(f"Predictor LoRA trainable parameters: {trainable}/{total}")

    if not bool(lora_kwargs.get("train_predictor", True)):
        for param in predictor.parameters():
            param.requires_grad = False
        trainable, total = count_trainable_parameters(predictor)
        logger.info(f"Predictor trainable parameters after freeze: {trainable}/{total}")


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    decay_encoder = [
        p
        for n, p in encoder.named_parameters()
        if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
    ]
    decay_predictor = [
        p
        for n, p in predictor.named_parameters()
        if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
    ]
    no_decay_encoder = [
        p
        for n, p in encoder.named_parameters()
        if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
    ]
    no_decay_predictor = [
        p
        for n, p in predictor.named_parameters()
        if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
    ]

    param_groups = []
    for params in (decay_encoder, decay_predictor):
        if params:
            param_groups.append({"params": params})
    for params in (no_decay_encoder, no_decay_predictor):
        if params:
            param_groups.append(
                {
                    "params": params,
                    "WD_exclude": zero_init_bias_wd,
                    "weight_decay": 0,
                }
            )
    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found for optimizer")

    if use_radamw:
        from src.utils.adamw import AdamW as RAdamW

        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
