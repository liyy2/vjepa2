# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    module_name,
    device,
    frames_per_clip,
    resolution,
    checkpoint,
    model_kwargs,
    wrapper_kwargs,
    trainable=False,
    trainable_blocks=0,
    trainable_patterns=None,
):
    """
    Build model and initialize from pretrained checkpoint.

    API requirements for Encoder module:
      1) Needs to be a pytorch module with 'forward()' function protocol:
        :param x: (Tensor) Video clip (shape=[batch_size x num_channels x num_frames x height x width])
        :returns: (Tensor) Representations of video clip (shape=[batch_size x num_encoder_tokens x feature_dim])
    """
    model = (
        importlib.import_module(f"{module_name}")
        .init_module(
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
        )
        .to(device)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if trainable:
        _make_encoder_trainable(
            model,
            trainable_blocks=trainable_blocks,
            trainable_patterns=trainable_patterns,
        )
    print(model)
    return model


def _make_encoder_trainable(model, trainable_blocks=0, trainable_patterns=None):
    trainable_blocks = int(trainable_blocks or 0)
    trainable_patterns = tuple(trainable_patterns or ())

    if trainable_blocks > 0:
        blocks = _find_transformer_blocks(model)
        if blocks is None:
            raise ValueError(
                "encoder_trainable_blocks was set, but no transformer blocks "
                "were found on the encoder wrapper."
            )
        start = max(0, len(blocks) - trainable_blocks)
        for block in blocks[start:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info("Unfroze encoder transformer blocks %d-%d of %d", start, len(blocks) - 1, len(blocks))

    if trainable_patterns:
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                param.requires_grad = True
        logger.info("Unfroze encoder parameters matching patterns: %s", trainable_patterns)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    if trainable_params <= 0:
        raise ValueError(
            "trainable encoder requested but no parameters were unfrozen. "
            "Set encoder_trainable_blocks or encoder_trainable_patterns."
        )
    logger.info(
        "Encoder trainable params: %.2fM / %.2fM",
        trainable_params / 1.0e6,
        total_params / 1.0e6,
    )


def _find_transformer_blocks(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "encoder", None),
        getattr(model, "backbone", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "blocks"):
            return candidate.blocks
    return None
