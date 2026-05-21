# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import logging
import csv
import json
import math
import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.video_classification_frozen.models import init_module
from evals.video_classification_frozen.utils import make_transforms
from src.heads.corn_head import CORNAttentiveClassifier
from src.heads.stats_head import (
    StatsPoolingClassifier,
    TemporalStatsPoolingClassifier,
    TemporalTransformerClassifier,
)
from src.datasets.data_manager import init_data
from src.losses.corn_loss import corn_expected_score, corn_loss
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, init_distributed
from src.utils.logging import AverageMeter, CSVLogger, WandBLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- VAL ONLY
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 12)

    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")
    args_wandb = args_eval.get("wandb", {}) or args_exp.get("wandb", {}) or {}

    # -- CLASSIFIER
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    head_type = args_classifier.get("head_type", "softmax")
    head_kwargs = args_classifier.get("head_kwargs", {}) or {}
    prediction_mode = args_classifier.get("prediction_mode", "argmax")
    selection_metric = args_classifier.get(
        "selection_metric",
        "quadratic_weighted_kappa" if head_type == "corn" else "val_acc",
    )

    # -- DATA
    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    num_classes = args_data.get("num_classes")
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    train_transform_kwargs = args_data.get("train_transform_kwargs", None)
    val_transform_kwargs = args_data.get("val_transform_kwargs", None)
    dataset_kwargs = args_data.get("dataset_kwargs", None)
    corn_pos_weight = None
    softmax_class_weight = None
    label_smoothing = 0.0
    if head_type == "corn":
        corn_pos_weight = _resolve_corn_pos_weight(
            args_classifier.get("corn_pos_weight"),
            train_data_path,
            num_classes,
        )
        if corn_pos_weight is not None:
            logger.info(f"Using CORN pos_weight: {corn_pos_weight.tolist()}")
    else:
        softmax_class_weight = _resolve_class_weight(
            args_classifier.get("class_weight"),
            train_data_path,
            num_classes,
        )
        label_smoothing = float(args_classifier.get("label_smoothing", 0.0) or 0.0)
        if softmax_class_weight is not None:
            logger.info(f"Using softmax class_weight: {softmax_class_weight.tolist()}")
        if label_smoothing:
            logger.info(f"Using softmax label_smoothing: {label_smoothing}")

    # -- OPTIMIZATION
    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    finetune_encoder = bool(args_opt.get("finetune_encoder", False))
    encoder_trainable_blocks = int(args_opt.get("encoder_trainable_blocks", 0) or 0)
    encoder_trainable_patterns = args_opt.get("encoder_trainable_patterns", []) or []
    encoder_opt_kwargs = None
    if finetune_encoder:
        if len(opt_kwargs) != 1:
            raise ValueError("finetune_encoder=true currently supports exactly one classifier/optimizer head.")
        encoder_lr = float(args_opt.get("encoder_lr", 1.0e-5))
        encoder_wd = float(args_opt.get("encoder_weight_decay", 0.01))
        encoder_opt_kwargs = dict(
            ref_wd=encoder_wd,
            final_wd=float(args_opt.get("encoder_final_weight_decay", encoder_wd)),
            start_lr=float(args_opt.get("encoder_start_lr", encoder_lr)),
            ref_lr=encoder_lr,
            final_lr=float(args_opt.get("encoder_final_lr", 0.0)),
            warmup=float(args_opt.get("encoder_warmup", 0.0) or 0.0),
        )
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    best_path = os.path.join(folder, "best.pt")
    metrics_path = os.path.join(folder, "metrics_latest.json")

    # -- make loggers
    if head_type in ("corn", "softmax", "stats", "temporal_stats", "temporal_transformer"):
        log_fields = (
            ("%d", "epoch"),
            ("%.5f", "train_acc"),
            ("%.5f", "val_acc"),
            ("%.5f", "val_spearman"),
            ("%.5f", "val_qwk"),
            ("%.5f", "val_mae"),
            ("%.5f", "train_coverage_rate"),
            ("%.5f", "train_temporal_span_rate"),
            ("%.5f", "val_coverage_rate"),
            ("%.5f", "val_temporal_span_rate"),
        )
    else:
        log_fields = (
            ("%d", "epoch"),
            ("%.5f", "train_acc"),
            ("%.5f", "val_acc"),
            ("%.5f", "train_coverage_rate"),
            ("%.5f", "train_temporal_span_rate"),
            ("%.5f", "val_coverage_rate"),
            ("%.5f", "val_temporal_span_rate"),
        )
    wandb_logger = None
    if rank == 0:
        csv_logger = CSVLogger(log_file, *log_fields)
        wandb_logger = _init_wandb_logger(args_wandb, log_fields, args_eval, folder, eval_tag)

    # Initialize model

    # -- init models
    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
        trainable=finetune_encoder,
        trainable_blocks=encoder_trainable_blocks,
        trainable_patterns=encoder_trainable_patterns,
    )
    encoder_trainable = any(p.requires_grad for p in encoder.parameters())
    # -- init classifier
    classifiers = [
        _build_classifier(
            head_type=head_type,
            embed_dim=encoder.embed_dim,
            num_heads=num_heads,
            depth=num_probe_blocks,
            num_classes=num_classes,
            head_kwargs=head_kwargs,
        ).to(device)
        for _ in opt_kwargs
    ]
    classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    if encoder_trainable:
        logger.info("Wrapping trainable encoder with DDP")
        encoder = DistributedDataParallel(encoder, static_graph=True)
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        normalization=normalization,
        transform_kwargs=train_transform_kwargs,
        dataset_kwargs=dataset_kwargs,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        normalization=normalization,
        transform_kwargs=val_transform_kwargs,
        dataset_kwargs=dataset_kwargs,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
        encoder=encoder if encoder_trainable else None,
        encoder_opt_kwargs=encoder_opt_kwargs,
    )

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        encoder, classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            encoder=encoder,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(epoch, path=latest_path, metrics=None):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "metrics": metrics,
            "selection_metric": selection_metric,
        }
        if encoder_trainable:
            save_dict["encoder"] = _trainable_encoder_state_dict(encoder)
            save_dict["encoder_trainable_only"] = True
        if rank == 0:
            torch.save(save_dict, path)

    # TRAIN LOOP
    best_metric = float("-inf")
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_acc = -1.0
            train_result = {"acc": train_acc}
        else:
            train_result = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                head_type=head_type,
                num_classes=num_classes,
                corn_pos_weight=corn_pos_weight,
                class_weight=softmax_class_weight,
                label_smoothing=label_smoothing,
                prediction_mode=prediction_mode,
                encoder_trainable=encoder_trainable,
            )
            train_acc = train_result["acc"]

        val_result = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            head_type=head_type,
            num_classes=num_classes,
            corn_pos_weight=corn_pos_weight,
            class_weight=softmax_class_weight,
            label_smoothing=label_smoothing,
            selection_metric=selection_metric,
            prediction_mode=prediction_mode,
            encoder_trainable=encoder_trainable,
        )
        val_acc = val_result["acc"]

        train_coverage_rate = _coverage_metric(train_result, "coverage_rate")
        train_temporal_span_rate = _coverage_metric(train_result, "temporal_span_rate")
        val_coverage_rate = _coverage_metric(val_result, "coverage_rate")
        val_temporal_span_rate = _coverage_metric(val_result, "temporal_span_rate")
        logger.info(
            "[%5d] train: %.3f%% test: %.3f%% coverage: train %.1f%%/%.1f%% val %.1f%%/%.1f%%"
            % (
                epoch + 1,
                train_acc,
                val_acc,
                100.0 * train_coverage_rate,
                100.0 * train_temporal_span_rate,
                100.0 * val_coverage_rate,
                100.0 * val_temporal_span_rate,
            )
        )
        if rank == 0:
            if head_type in ("corn", "softmax", "stats", "temporal_stats", "temporal_transformer"):
                log_values = (
                    epoch + 1,
                    train_acc,
                    val_acc,
                    val_result.get("spearman", float("nan")),
                    val_result.get("quadratic_weighted_kappa", float("nan")),
                    val_result.get("mae", float("nan")),
                    train_coverage_rate,
                    train_temporal_span_rate,
                    val_coverage_rate,
                    val_temporal_span_rate,
                )
                csv_logger.log(*log_values)
                if wandb_logger is not None:
                    wandb_logger.log(*log_values)
                with open(metrics_path, "w") as handle:
                    json.dump(
                        {
                            "epoch": epoch + 1,
                            "train": train_result if not val_only else {"acc": train_acc},
                            "val": val_result,
                            "selection_metric": selection_metric,
                        },
                        handle,
                        indent=2,
                    )
            else:
                log_values = (
                    epoch + 1,
                    train_acc,
                    val_acc,
                    train_coverage_rate,
                    train_temporal_span_rate,
                    val_coverage_rate,
                    val_temporal_span_rate,
                )
                csv_logger.log(*log_values)
                if wandb_logger is not None:
                    wandb_logger.log(*log_values)

        if val_only:
            if rank == 0 and wandb_logger is not None:
                wandb_logger.finish()
            return

        save_checkpoint(epoch + 1, path=latest_path, metrics=val_result)
        metric_value = val_result.get(selection_metric, val_acc)
        if isinstance(metric_value, (int, float)) and np.isfinite(metric_value) and metric_value > best_metric:
            best_metric = metric_value
            save_checkpoint(epoch + 1, path=best_path, metrics=val_result)
    if rank == 0 and wandb_logger is not None:
        wandb_logger.finish()


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    head_type="softmax",
    num_classes=None,
    corn_pos_weight=None,
    class_weight=None,
    label_smoothing=0.0,
    selection_metric=None,
    prediction_mode="argmax",
    encoder_trainable=False,
):

    for c in classifiers:
        c.train(mode=training)
    encoder.train(mode=training and encoder_trainable)

    if head_type == "corn":
        if corn_pos_weight is not None:
            corn_pos_weight = corn_pos_weight.to(device)
        criterion = lambda logits, labels: corn_loss(
            logits,
            labels,
            num_levels=num_classes,
            pos_weight=corn_pos_weight,
        )
    else:
        if class_weight is not None:
            class_weight = class_weight.to(device)
        label_smoothing = float(label_smoothing or 0.0)
        criterion = lambda logits, labels: F.cross_entropy(
            logits,
            labels,
            weight=class_weight.to(dtype=logits.dtype) if class_weight is not None else None,
            label_smoothing=label_smoothing,
        )
    top1_meters = [AverageMeter() for _ in classifiers]
    coverage_meters = {key: AverageMeter() for key in ("coverage_rate", "temporal_span_rate")}
    ordinal_scores = [[] for _ in classifiers]
    ordinal_preds = [[] for _ in classifiers]
    ordinal_labels = [[] for _ in classifiers]
    ordinal_indices = [[] for _ in classifiers]
    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
            # Load data and put on GPU
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                for di in data[0]  # iterate over temporal index of clip
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            metadata_features = data[4].to(device, non_blocking=True) if len(data) > 4 else None
            batch_size = len(labels)
            batch_coverage = _batch_coverage(data[3]) if len(data) > 3 else {}
            batch_sample_indices = _batch_sample_indices(data[3], batch_size) if len(data) > 3 else None
            for key, meter in coverage_meters.items():
                if key in batch_coverage:
                    meter.update(float(AllReduce.apply(torch.tensor(batch_coverage[key], device=device))), batch_size)

            # Forward and prediction
            if training and encoder_trainable:
                encoder_outputs = encoder(clips, clip_indices)
            else:
                with torch.no_grad():
                    encoder_outputs = encoder(clips, clip_indices)
            if training:
                outputs = [
                    [_classifier_forward(c, o, metadata_features) for o in encoder_outputs]
                    for c in classifiers
                ]
            else:
                with torch.no_grad():
                    outputs = [
                        [_classifier_forward(c, o, metadata_features) for o in encoder_outputs]
                        for c in classifiers
                    ]

        # Compute loss
        losses = [[criterion(o, labels) for o in coutputs] for coutputs in outputs]
        with torch.no_grad():
            if head_type == "corn":
                outputs = [sum([corn_expected_score(o) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                pred_outputs = [coutputs.round().clamp(0, num_classes - 1).long() for coutputs in outputs]
                top1_accs = [
                    100.0 * pred_outputs_i.eq(labels).sum() / batch_size
                    for pred_outputs_i in pred_outputs
                ]
                if not training:
                    labels_cpu = labels.detach().cpu().long().numpy().tolist()
                    for classifier_idx, (scores_i, preds_i) in enumerate(zip(outputs, pred_outputs)):
                        ordinal_scores[classifier_idx].extend(scores_i.detach().cpu().float().numpy().tolist())
                        ordinal_preds[classifier_idx].extend(preds_i.detach().cpu().long().numpy().tolist())
                        ordinal_labels[classifier_idx].extend(labels_cpu)
                        if batch_sample_indices is not None:
                            ordinal_indices[classifier_idx].extend(batch_sample_indices)
            else:
                outputs = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                score_range = torch.arange(num_classes, device=device, dtype=outputs[0].dtype)
                score_outputs = [(probs_i * score_range).sum(dim=1) for probs_i in outputs]
                if prediction_mode == "expected_round":
                    pred_outputs = [
                        scores_i.round().clamp(0, num_classes - 1).long()
                        for scores_i in score_outputs
                    ]
                else:
                    pred_outputs = [coutputs.max(dim=1).indices for coutputs in outputs]
                top1_accs = [
                    100.0 * pred_outputs_i.eq(labels).sum() / batch_size
                    for pred_outputs_i in pred_outputs
                ]
                if not training:
                    labels_cpu = labels.detach().cpu().long().numpy().tolist()
                    for classifier_idx, (scores_i, preds_i) in enumerate(zip(score_outputs, pred_outputs)):
                        ordinal_scores[classifier_idx].extend(scores_i.detach().cpu().float().numpy().tolist())
                        ordinal_preds[classifier_idx].extend(preds_i.detach().cpu().long().numpy().tolist())
                        ordinal_labels[classifier_idx].extend(labels_cpu)
                        if batch_sample_indices is not None:
                            ordinal_indices[classifier_idx].extend(batch_sample_indices)
            top1_accs = [float(AllReduce.apply(t1a)) for t1a in top1_accs]
            for t1m, t1a in zip(top1_meters, top1_accs):
                t1m.update(t1a)

        if training:
            if use_bfloat16:
                [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                [s.step(o) for s, o in zip(scaler, optimizer)]
                [s.update() for s in scaler]
            else:
                [[lij.backward() for lij in li] for li in losses]
                [o.step() for o in optimizer]
            [o.zero_grad() for o in optimizer]

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        if itr % 10 == 0:
            logger.info(
                "[%5d] %.3f%% [%.3f%% %.3f%%] [mem: %.2e]"
                % (
                    itr,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                )
            )

    result = {
        "acc": float(_agg_top1.max()),
        "coverage": {key: float(meter.avg) for key, meter in coverage_meters.items() if meter.count > 0},
    }
    if head_type in ("corn", "softmax", "stats", "temporal_stats", "temporal_transformer") and not training:
        per_classifier = []
        for scores_i, preds_i, labels_i, indices_i in zip(
            ordinal_scores,
            ordinal_preds,
            ordinal_labels,
            ordinal_indices,
        ):
            gathered_scores = _all_gather_python_list(scores_i)
            gathered_preds = _all_gather_python_list(preds_i)
            gathered_labels = _all_gather_python_list(labels_i)
            gathered_indices = _all_gather_python_list(indices_i) if indices_i else None
            per_classifier.append(
                _ordinal_metrics(
                    labels=np.array(gathered_labels, dtype=np.int64),
                    predictions=np.array(gathered_preds, dtype=np.int64),
                    scores=np.array(gathered_scores, dtype=np.float64),
                    sample_indices=(
                        np.array(gathered_indices, dtype=np.int64)
                        if gathered_indices is not None
                        else None
                    ),
                    num_classes=num_classes,
                )
            )
        best_idx = _best_metric_index(
            per_classifier,
            selection_metric or ("quadratic_weighted_kappa" if head_type == "corn" else "accuracy"),
        )
        result.update(per_classifier[best_idx])
        result["acc"] = result.get("val_acc", result["acc"])
        result["best_classifier"] = int(best_idx)
        result["per_classifier"] = per_classifier
    return result


def _build_classifier(head_type, embed_dim, num_heads, depth, num_classes, head_kwargs=None):
    head_kwargs = head_kwargs or {}
    if head_type == "corn":
        return CORNAttentiveClassifier(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            num_levels=num_classes,
            use_activation_checkpointing=True,
        )
    if head_type == "stats":
        return StatsPoolingClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            **head_kwargs,
        )
    if head_type == "temporal_stats":
        return TemporalStatsPoolingClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            **head_kwargs,
        )
    if head_type == "temporal_transformer":
        return TemporalTransformerClassifier(
            embed_dim=embed_dim,
            num_classes=num_classes,
            **head_kwargs,
        )
    return AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_classes=num_classes,
        use_activation_checkpointing=True,
    )


def _classifier_forward(classifier, tokens, metadata_features=None):
    # Unwrap DDP / FSDP so `metadata_dim` lookup hits the actual module, not
    # the wrapper. Without this, DDP-wrapped heads silently lose metadata.
    inner = getattr(classifier, "module", classifier)
    if metadata_features is not None and getattr(inner, "metadata_dim", 0) > 0:
        return classifier(tokens, metadata_features)
    return classifier(tokens)


def _init_wandb_logger(wandb_cfg, log_fields, args_eval, folder, eval_tag):
    if not wandb_cfg.get("enabled", False):
        return None
    wandb_dir = wandb_cfg.get("dir", folder)
    os.makedirs(wandb_dir, exist_ok=True)
    name = wandb_cfg.get("name") or eval_tag or os.path.basename(folder.rstrip(os.sep))
    project = wandb_cfg.get("project", "vjepa2")
    logger.info(f"Initializing wandb logger project={project} name={name}")
    return WandBLogger(
        *log_fields,
        enabled=True,
        project=project,
        entity=wandb_cfg.get("entity"),
        name=name,
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        mode=wandb_cfg.get("mode"),
        dir=wandb_dir,
        resume=wandb_cfg.get("resume"),
        id=wandb_cfg.get("id"),
        job_type=wandb_cfg.get("job_type", "video_classification_frozen"),
        config=wandb_cfg.get("config", args_eval),
    )


def _resolve_corn_pos_weight(config_value, train_paths, num_classes):
    if config_value in (None, False):
        return None
    if isinstance(config_value, str) and config_value.lower() == "auto":
        labels = _read_labels_from_csvs(train_paths)
        if not labels:
            raise ValueError("corn_pos_weight=auto requires labels in the train CSV.")
        weights = []
        label_array = np.array(labels, dtype=np.int64)
        for threshold in range(num_classes - 1):
            positives = int(np.sum(label_array > threshold))
            negatives = int(np.sum(label_array <= threshold))
            weights.append(float(negatives / positives) if positives > 0 else 1.0)
        logger.info(
            "Auto CORN threshold counts: %s",
            [
                {
                    "threshold": threshold,
                    "positive": int(np.sum(label_array > threshold)),
                    "negative": int(np.sum(label_array <= threshold)),
                }
                for threshold in range(num_classes - 1)
            ],
        )
        return torch.tensor(weights, dtype=torch.float32)
    if isinstance(config_value, (list, tuple)):
        weights = [float(value) for value in config_value]
        if len(weights) != num_classes - 1:
            raise ValueError(f"Expected {num_classes - 1} CORN pos weights, got {len(weights)}.")
        return torch.tensor(weights, dtype=torch.float32)
    raise ValueError("corn_pos_weight must be 'auto', a list of weights, false, or omitted.")


def _resolve_class_weight(config_value, train_paths, num_classes):
    if config_value in (None, False):
        return None
    if isinstance(config_value, str) and config_value.lower() in ("auto", "balanced"):
        labels = _read_labels_from_csvs(train_paths)
        if not labels:
            raise ValueError("class_weight=auto requires labels in the train CSV.")
        counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes)
        total = float(counts.sum())
        weights = [
            total / (num_classes * float(count)) if count > 0 else 0.0
            for count in counts[:num_classes]
        ]
        logger.info(
            "Auto softmax class counts: %s",
            {int(idx): int(count) for idx, count in enumerate(counts[:num_classes])},
        )
        return torch.tensor(weights, dtype=torch.float32)
    if isinstance(config_value, (list, tuple)):
        weights = [float(value) for value in config_value]
        if len(weights) != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {len(weights)}.")
        return torch.tensor(weights, dtype=torch.float32)
    raise ValueError("class_weight must be 'auto'/'balanced', a list of weights, false, or omitted.")


def _read_labels_from_csvs(paths):
    labels = []
    for path in paths:
        if not path:
            continue
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            label_col = "label" if "label" in reader.fieldnames else reader.fieldnames[-1]
            for row in reader:
                labels.append(int(float(row[label_col])))
    return labels


def _batch_coverage(coverage):
    if not isinstance(coverage, dict):
        return {}
    metrics = {}
    for key in ("coverage_rate", "temporal_span_rate"):
        value = coverage.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            metrics[key] = float(value.float().mean().item())
        elif isinstance(value, (list, tuple)) and value:
            metrics[key] = float(np.mean(value))
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _batch_sample_indices(coverage, batch_size):
    if not isinstance(coverage, dict):
        return None
    value = coverage.get("sample_index")
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        indices = value.detach().cpu().long().numpy().tolist()
    elif isinstance(value, (list, tuple)):
        indices = [int(v) for v in value]
    elif isinstance(value, (int, float)):
        indices = [int(value)]
    else:
        return None
    if len(indices) != batch_size:
        return None
    return indices


def _coverage_metric(result, key):
    value = result.get("coverage", {}).get(key, float("nan"))
    return value if isinstance(value, (int, float)) and np.isfinite(value) else float("nan")


def _all_gather_python_list(values):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, list(values))
        flat = []
        for item in gathered:
            flat.extend(item)
        return flat
    return list(values)


def _best_metric_index(metrics, key):
    if key == "val_acc":
        key = "accuracy"
    values = [m.get(key, float("nan")) for m in metrics]
    finite = [v if isinstance(v, (int, float)) and np.isfinite(v) else float("-inf") for v in values]
    return int(np.argmax(finite))


def _ordinal_metrics(labels, predictions, scores, num_classes, sample_indices=None):
    raw_n = int(labels.size)
    if sample_indices is not None and sample_indices.size == labels.size:
        keep = []
        seen = set()
        for row_idx, sample_idx in enumerate(sample_indices.tolist()):
            sample_idx = int(sample_idx)
            if sample_idx in seen:
                continue
            seen.add(sample_idx)
            keep.append(row_idx)
        keep = np.array(keep, dtype=np.int64)
        labels = labels[keep]
        predictions = predictions[keep]
        scores = scores[keep]
        sample_indices = sample_indices[keep]
        order = np.argsort(sample_indices, kind="mergesort")
        labels = labels[order]
        predictions = predictions[order]
        scores = scores[order]

    if labels.size == 0:
        return {
            "n": 0,
            "raw_n": raw_n,
            "accuracy": float("nan"),
            "val_acc": float("nan"),
            "spearman": float("nan"),
            "quadratic_weighted_kappa": float("nan"),
            "mae": float("nan"),
            "confusion_matrix": [[0 for _ in range(num_classes)] for _ in range(num_classes)],
        }
    predictions = np.clip(predictions, 0, num_classes - 1)
    labels = np.clip(labels, 0, num_classes - 1)
    return {
        "n": int(labels.size),
        "raw_n": raw_n,
        "accuracy": float(100.0 * np.mean(predictions == labels)),
        "val_acc": float(100.0 * np.mean(predictions == labels)),
        "spearman": _spearman(labels.astype(float), scores.astype(float)),
        "quadratic_weighted_kappa": _quadratic_weighted_kappa(labels, predictions, num_classes),
        "mae": float(np.mean(np.abs(predictions - labels))),
        "confusion_matrix": _confusion_matrix(labels, predictions, num_classes).tolist(),
    }


def _rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(labels, scores):
    if len(labels) < 2:
        return float("nan")
    labels_rank = _rankdata(labels)
    scores_rank = _rankdata(scores)
    if np.std(labels_rank) == 0 or np.std(scores_rank) == 0:
        return float("nan")
    return float(np.corrcoef(labels_rank, scores_rank)[0, 1])


def _quadratic_weighted_kappa(labels, predictions, num_classes):
    observed = _confusion_matrix(labels, predictions, num_classes).astype(np.float64)
    total = observed.sum()
    if total == 0:
        return float("nan")
    label_hist = observed.sum(axis=1)
    pred_hist = observed.sum(axis=0)
    expected = np.outer(label_hist, pred_hist) / total
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    denom = float((num_classes - 1) ** 2)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / denom
    observed_weighted = (weights * observed).sum()
    expected_weighted = (weights * expected).sum()
    if expected_weighted == 0:
        return float("nan")
    return float(1.0 - observed_weighted / expected_weighted)


def _confusion_matrix(labels, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions):
        matrix[int(label), int(prediction)] += 1
    return matrix


def _unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def _trainable_encoder_state_dict(encoder):
    module = _unwrap_module(encoder)
    trainable_names = {name for name, param in module.named_parameters() if param.requires_grad}
    state = module.state_dict()
    return {
        name: tensor.detach().cpu()
        for name, tensor in state.items()
        if name in trainable_names
    }


def load_checkpoint(device, r_path, encoder, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    if encoder is not None and checkpoint.get("encoder") is not None:
        msg = _unwrap_module(encoder).load_state_dict(checkpoint["encoder"], strict=False)
        logger.info(f"loaded trainable encoder state with msg: {msg}")

    # -- loading encoder
    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return encoder, classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return encoder, classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None,
    normalization=None,
    transform_kwargs=None,
    dataset_kwargs=None,
):
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    # Make Video Transforms
    transform_args = dict(
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )
    transform_args.update(transform_kwargs or {})
    for tuple_key in ("random_resize_aspect_ratio", "random_resize_scale"):
        if isinstance(transform_args.get(tuple_key), list):
            transform_args[tuple_key] = tuple(transform_args[tuple_key])
    transform = make_transforms(
        training=training,
        **transform_args,
    )

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=training,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        random_clip_sampling=training,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        drop_last=False,
        subset_file=subset_file,
        dataset_kwargs=dataset_kwargs,
    )
    return data_loader, data_sampler


def init_opt(
    classifiers,
    iterations_per_epoch,
    opt_kwargs,
    num_epochs,
    use_bfloat16=False,
    encoder=None,
    encoder_opt_kwargs=None,
):
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": [p for p in c.parameters() if p.requires_grad],
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        if encoder is not None and encoder_opt_kwargs is not None:
            encoder_params = [p for p in _unwrap_module(encoder).parameters() if p.requires_grad]
            if len(encoder_params) == 0:
                raise ValueError("encoder_opt_kwargs was provided, but the encoder has no trainable parameters.")
            param_groups.append(
                {
                    "params": encoder_params,
                    "mc_warmup_steps": int(encoder_opt_kwargs.get("warmup") * iterations_per_epoch),
                    "mc_start_lr": encoder_opt_kwargs.get("start_lr"),
                    "mc_ref_lr": encoder_opt_kwargs.get("ref_lr"),
                    "mc_final_lr": encoder_opt_kwargs.get("final_lr"),
                    "mc_ref_wd": encoder_opt_kwargs.get("ref_wd"),
                    "mc_final_wd": encoder_opt_kwargs.get("final_wd"),
                }
            )
            logger.info("Optimizer includes %.2fM trainable encoder parameters", sum(p.numel() for p in encoder_params) / 1.0e6)
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        scalers += [torch.cuda.amp.GradScaler() if use_bfloat16 else None]
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                # -- progress after warmup
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
