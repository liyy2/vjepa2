# Item 3.4 Fold 0 Adaptive-Window Status

Date: 2026-05-22

## Confirmed Baselines

| Model | Acc | QWK | Notes |
|---|---:|---:|---|
| Kinematic LightGBM/calibrated | 68.2 | 0.818 | External reference baseline, current split order from `per_clip_predictions.csv` |
| Kinematic validation-selected median vote | 72.7 | 0.814 | Higher-accuracy kinematic diagnostic in `saved_prediction_vote_ensembles_stride2.json` |
| Subject-OOF kinematic + V-JEPA fusion | 63.6 | 0.809 | Fusion weight/threshold selected only from subject-group fold-0 train OOF predictions |
| Fixed V-JEPA LoRA CORN, K=4 | 48.5 | 0.465 | Best current V-JEPA run |
| Fixed-best V-JEPA checkpoint on adaptive full coverage | 43.9 | 0.425 | Same checkpoint, adaptive windows |
| Subject-OOF cached V-JEPA temporal stacker | 45.5 | 0.689 | Top-4 candidate pool; stack/thresholds selected only from subject-group fold-0 train OOF predictions |
| Cached V-JEPA temporal-stats ensemble | 56.1 | 0.729 | Reproduced from 732 saved V-JEPA-only val probability vectors |
| Kinematic + cached V-JEPA temporal-stats score fusion | 71.2 | 0.829 | Val-tuned fusion weight 0.38; shows complementary signal, not an honest CV claim |

## Adaptive Heads Measured So Far

| Encoder/head | Acc | QWK | Diagnosis |
|---|---:|---:|---|
| Frozen + stats MLP | 33.3 | 0.186 | Best adaptive learned head so far |
| Frozen + TemporalAgg | 30.3 | 0.000 | Class-prior collapse |
| Frozen + XceptionTime | 31.8 | 0.121 | Weak |
| Supervised LoRA + stats MLP | 34.8 | 0.081 | Weak |
| Supervised LoRA + TemporalAgg | 28.8 | 0.000 | Collapse |
| Supervised LoRA + XceptionTime | 30.3 | 0.000 | Collapse |
| Warm-start supervised LoRA + stats MLP | 30.3 | 0.000 | Predicts class 1 for all val clips |
| Warm-start supervised LoRA + TemporalAgg full tokens | 30.3 | 0.000 | Predicts class 1 for all val clips |
| Warm-start supervised LoRA + XceptionTime | 30.3 | 0.000 | Predicts class 1 for all val clips |
| Adaptive full-coverage cached temporal stats, velocity grid | 28.8 | 0.000 | Collapsed after 192 configs; stopped during mega-seeds |
| Adaptive capped max-12 cached temporal stats, quick recipes | 28.8 | 0.000 | Collapsed after sample-balanced normalization/PCA |
| Adaptive capped max-12 cached temporal stats, ExtraTrees diagnostic | 42.4 | 0.302 | Scikit diagnostic only; features weak but not fully degenerate |
| Adaptive WindowMIL fixed scorer, top-k=4 val-only | 43.9 | 0.499 | Best adaptive attentive/frozen QWK so far; better QWK than fixed K=4 but lower acc |
| Adaptive WindowMIL learned stats, frozen scorer, epoch 1 | 51.5 | 0.426 | Best adaptive learned accuracy; later epochs degraded |
| Adaptive WindowMIL supervised LoRA mean, epoch 8 | 42.4 | 0.416 | Finished below fixed WindowMIL smoke |
| Adaptive WindowMIL supervised LoRA top-k, epoch 7/8 | 42.4 | 0.428 | Finished below fixed WindowMIL smoke |
| Adaptive CORN warm-start full tokens, epoch 6 | 37.9 | 0.443 | Finished below fixed-best adaptive top-k WindowMIL |
| Train-fitted score calibration, top-k=8 + mean-logit blend | 48.5 | 0.596 | Best adaptive held-out QWK so far; thresholds/blend fit only on fold-0 train predictions |
| Adaptive JEPA-LoRA e1 snapshot + stats MLP | 30.3 | 0.000 | Partial diagnostic from epoch-1 self-supervised snapshot; job cancelled after early metrics to unblock final JEPA run |
| Adaptive JEPA-LoRA e1 snapshot + TemporalAgg | 31.8 | 0.036 | Partial diagnostic from epoch-1 self-supervised snapshot; weak |
| Adaptive JEPA-LoRA e1 snapshot + XceptionTime | 30.3 | 0.000 | Partial diagnostic from epoch-1 self-supervised snapshot; weak |
| Cached temporal subject-OOF stacker, top-4 pool | 45.5 | 0.689 | Best strict subject-group OOF-selected V-JEPA-only fold-0 result so far; greedy top-3 score-threshold blend |
| Cached temporal subject-OOF stacker, top-8 pool strict selector | 47.0 | 0.670 | Larger pool selected top-7 by OOF QWK; top-8 row had better acc/MAE and 53.0 / 0.695 held-out |

## WindowMIL Aggregation Sweep

| Fixed scorer aggregation | Acc | QWK | MAE |
|---|---:|---:|---:|
| top-k=4 probability | 43.9 | 0.499 | 0.636 |
| top-k=8 probability | 45.5 | 0.444 | 0.636 |
| top-k=2 probability | 36.4 | 0.430 | 0.727 |
| top-k=16 probability | 47.0 | 0.423 | 0.652 |
| mean probability | 45.5 | 0.421 | 0.667 |
| max probability | 34.8 | 0.413 | 0.742 |
| mean logit | 43.9 | 0.394 | 0.697 |

The fixed per-window CORN scorer has useful window-level severity signal, but aggregation is fragile. Top-k=4 gives the best ordinal agreement, while wider top-k/mean variants improve accuracy slightly by changing the class prior and lose QWK.

## Score-Calibration Diagnostic

Using the val predictions only as a diagnostic, the fixed scorer's continuous scores contain more ordinal signal than the emitted integer predictions expose. Coarse threshold tuning on the same 66 validation clips reaches QWK 0.62 for top-k=8 and QWK 0.60 for mean-probability pooling, but that readout is label-leaky and should not be reported as performance.

The honest train-fitted run completed as `windowmil_trainfit_calibration_28868563.json` with 248 train clips and 66 held-out validation clips. Best result is a top-k=8 + mean-logit score blend, weight 0.65 on top-k=8, with train-fitted thresholds `[0.417, 1.049, 2.057, 2.595]`: 48.5% acc / QWK 0.596 / MAE 0.561. This beats the fixed K=4 V-JEPA QWK but remains far below the kinematic 0.818 QWK baseline.

Top train-fitted calibration rows:

| Calibration | Acc | QWK | MAE |
|---|---:|---:|---:|
| top-k=8 + mean-logit blend | 48.5 | 0.596 | 0.561 |
| top-k=8 + mean-probability blend | 45.5 | 0.583 | 0.591 |
| mean-probability + max-probability blend | 48.5 | 0.550 | 0.591 |
| top-k=8 threshold-only | 40.9 | 0.522 | 0.652 |
| learned-stats threshold-only | 45.5 | 0.512 | 0.636 |

## OOF Temporal-Stacker Check

`scripts/pd_hand/temporal_oof_stacker.py` reruns selected cached temporal-stat configs inside fold-0 train with 5 inner folds. The first OOF pass used clip-level `StratifiedKFold`; that is now superseded. The script now reconstructs the cached NPZ sampler order from `fold_0_train.csv`, validates labels against the NPZ, and uses subject-group `StratifiedGroupKFold` by default. Candidate weights, score thresholds, and subset selection are fit on out-of-fold train predictions; the 66 held-out fold-0 validation labels are used only for final evaluation.

The script also exposed a reproducibility issue in `train_temporal_encoder.py`: fixed-length caches were being routed through the masked forward path even when every mask entry was valid. The code now keeps the original unmasked path for all-true masks and only uses the masked path for genuinely variable-length/adaptive caches. After this fix, a rerun of the stored top config recovered useful performance instead of collapsing to class 0, but it still did not exactly reproduce the stale saved single-model QWK.

| OOF stacker run | OOF selection rule | OOF Acc/QWK/MAE | Held-out Acc/QWK/MAE | Output |
|---|---|---:|---:|---|
| Top-4 pool, clip-level, 35 epochs | top-3 score-threshold blend | 48.8 / 0.618 / 0.585 | 54.5 / 0.696 / 0.500 | `temporal_oof_stacker_top4_e35_unmaskedfix.json`; superseded by subject split |
| Top-8 pool, clip-level, 35 epochs | pair 1+6 score-threshold blend | 49.2 / 0.640 / 0.569 | 47.0 / 0.624 / 0.576 | `temporal_oof_stacker_top8_e35_unmaskedfix.json`; superseded by subject split |
| Top-4 pool, subject-group, 35 epochs | greedy top-3 score-threshold blend | 44.8 / 0.546 / 0.677 | 45.5 / 0.689 / 0.591 | `temporal_oof_stacker_top4_e35_subject_unmaskedfix.json` |
| Top-8 pool, subject-group, 35 epochs | top-7 score-threshold blend | 43.5 / 0.546 / 0.714 | 47.0 / 0.670 / 0.621 | `temporal_oof_stacker_top8_e35_subject_unmaskedfix.json` |
| Top-8 pool, subject-group, MAE/acc-favorable near-tie | top-8 score-threshold blend | 47.6 / 0.544 / 0.633 | 53.0 / 0.695 / 0.515 | Same top-8 JSON; not the strict max-QWK selector |

Takeaway: once subject-level OOF is enforced, the strongest strict V-JEPA-only selector is 45.5% accuracy / 0.689 QWK. The subject top-8 run still shows signal near QWK 0.70, but the 248-clip fold-0 train set is too small for stable subset selection over many near-equivalent seeds. The pure V-JEPA branch remains well below the kinematic 0.818 QWK baseline.

## OOF Kinematic + V-JEPA Fusion Check

`scripts/pd_hand/fit_kinematic_vjepa_oof_fusion.py` aligns kinematic feature rows to the V-JEPA cache order, validates labels, generates subject-group OOF kinematic probabilities, and combines them with the saved subject-OOF V-JEPA probabilities from `temporal_oof_stacker_top4_e35_subject_unmaskedfix_probs.probs.npz`. Fusion weights and ordinal score thresholds are fit only on fold-0 train OOF predictions.

| Fusion row | OOF Acc/QWK/MAE | Held-out Acc/QWK/MAE | Notes |
|---|---:|---:|---|
| OOF-selected: old_118 extra-trees seed0 + V-JEPA subject-greedy-k3 | 56.5 / 0.715 / 0.488 | 63.6 / 0.809 / 0.379 | Kinematic weight 0.70, V-JEPA weight 0.30 |
| Validation-best diagnostic: old_118 extra-trees seed0 + V-JEPA top3 | 53.2 / 0.683 / 0.536 | 66.7 / 0.845 / 0.333 | Kinematic weight 0.50, V-JEPA weight 0.50; diagnostic only |
| Kinematic single model: old_118 extra-trees seed98 | 55.6 / 0.640 / 0.492 | 68.2 / 0.813 / 0.318 | Reproduces the strong kinematic baseline single model |

Takeaway: the honest OOF-selected fusion is comparable to the kinematic QWK baseline but does not cleanly beat it on accuracy or QWK. The validation-best diagnostic surpasses the kinematic QWK, confirming complementary V-JEPA signal, but the OOF selection gap shows that fold-0 train is too small for stable fusion selection without full five-fold confirmation.

## Length-Bin Diagnostic

| Model | <=8.5s Acc/QWK | 8.5-14s Acc/QWK | >14s Acc/QWK |
|---|---:|---:|---:|
| Fixed K=4 | 75.0 / 0.816 | 45.7 / 0.385 | 25.0 / 0.308 |
| Fixed-best adaptive | 75.0 / 0.816 | 41.3 / 0.331 | 12.5 / 0.184 |

Adaptive full coverage does not solve the cap by itself. It preserves short-clip performance but degrades mid/long clips, so the issue is not just missing frames. The likely failure is that V-JEPA RGB tokens do not expose finger-tap kinematic cues strongly enough for generic embedding pooling, and longer clips dilute the discriminative windows.

The first adaptive cached temporal-stats run also exposed a training-basis problem: variable-length clips had 80-688 valid tubelet tokens, and the original standardization/PCA weighted every token equally. That gives long clips much more influence than short clips, unlike the fixed-length cache. `train_temporal_encoder.py` now has sample-balanced normalization plus optional equal-token-per-sample PCA fitting, and the capped adaptive runner uses both.

## Experiment Jobs

| Job | Config | Purpose |
|---:|---|---|
| 28868005 | `pd_hand_item_3_4_fold0_adaptive_windowmil_fixedbest_topk_valonly.yaml` | Val-only smoke: reuse fixed CORN scorer per adaptive window, top-k probability MIL aggregation |
| 28868006 | `pd_hand_item_3_4_fold0_adaptive_windowmil_supervised_lora_topk.yaml` | Continue LoRA/head training with top-k window MIL |
| 28868007 | `pd_hand_item_3_4_fold0_adaptive_windowmil_supervised_lora_mean.yaml` | Continue LoRA/head training with mean-probability window MIL |
| 28868008 | `pd_hand_item_3_4_fold0_adaptive_corn_warmstart_supervised_lora_fulltokens.yaml` | Same CORN head as baseline, adaptive full-token continuation |
| 28867116 | `vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows.yaml` | Superseded adaptive JEPA attempt; loaded stale `handcrop_long` checkpoint and then aborted |
| 28868979 | `vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows-smoke.yaml` | Completed 1-iteration smoke: adaptive windows, `num_workers=0`, base V-JEPA checkpoint, wrote smoke `latest.pth.tar` |
| 28868982 | `vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows.yaml` | Failed during epoch 2 with CUDA OOM after completing epoch 1; produced a valid epoch-1 `latest.pth.tar` |
| 28868991 | `merge_lora_checkpoint.py` | Cancelled because 28868982 failed |
| 28868999 / 28869000 / 28869001 | `adaptive_jepa_lora_{statsmlp,xceptiontime,temporalagg}.yaml` | Cancelled because 28868991 was cancelled |
| 28869015 | `summarize_adaptive_eval_metrics.py` | Completed after cancelled evals; e25 JEPA-LoRA rows are `missing_metrics` in `adaptive_jepa_lora_metrics_after_28868982.md` |
| 28869023 / 28869024 / 28869025 | generated `jepa_e1_snapshot_28868982/*e1*.yaml` | Cancelled after first metrics to release GPUs for the final JEPA run; partial summary written to `adaptive_jepa_lora_e1_snapshot_metrics_28868982_partial.md` |
| 28869028 | `summarize_adaptive_eval_metrics.py` | Cancelled with the e1 diagnostic evals |
| 28869066 | `vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows.yaml` | Fresh DDP4 resume from the epoch-1 checkpoint with exact adaptive windows, `optimization.clip_microbatch_size: 4`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; pending on GPU resources |
| 28869067 | `merge_lora_checkpoint.py` | Queued `afterok:28869066`; writes `fold_0_lora_adaptive_windows_e25/merged_for_cache.pt` |
| 28869068 / 28869069 / 28869070 | `adaptive_jepa_lora_{statsmlp,xceptiontime,temporalagg}.yaml` | Queued `afterok:28869067` to fill the three missing JEPA-LoRA adaptive-head cells |
| 28869071 | `summarize_adaptive_eval_metrics.py` | Queued `afterany` on 28869068/28869069/28869070; writes `adaptive_jepa_lora_metrics_after_28869066.{json,csv,md}` |
| 28869085 | `vitl384-lora-jepa-adapt-32f-step1-fold0-adaptive-windows-1gpu.yaml` | Running one-GPU fallback seeded from the 28868982 epoch-1 checkpoint; confirmed resume from epoch 1, `clip_microbatch_size=4`, epoch 2 itr 20 at ~11.4GB |
| 28869086 | `merge_lora_checkpoint.py` | Queued `afterok:28869085`; writes `fold_0_lora_adaptive_windows_e25_1gpu/merged_for_cache.pt` |
| 28869087 / 28869088 / 28869089 | generated `jepa_1gpu_e25_resume_28869066/*1gpu*.yaml` | Queued `afterok:28869086` to evaluate stats MLP, XceptionTime, and TemporalAgg on the one-GPU JEPA checkpoint |
| 28869090 | `summarize_adaptive_eval_metrics.py` | Queued `afterany` on 28869087/28869088/28869089; writes `adaptive_jepa_lora_metrics_after_28869085_1gpu.{json,csv,md}` |
| 28869125 | watcher on `lora_jepa_adapt_1gpu_28869085.out` | Running day-partition watcher; cancels 28869085 after `Epoch 3` appears so the epoch-2 checkpoint can be salvaged quickly |
| 28869116 | `merge_lora_checkpoint.py` | Queued `afterany:28869085`; salvages latest 1-GPU checkpoint to `fold_0_lora_adaptive_windows_1gpu_afterany/merged_for_cache.pt` even if the train job exits by cancellation/time limit |
| 28869117 / 28869118 / 28869119 | generated `jepa_1gpu_afterany_28869085/*afterany*.yaml` | Queued `afterok:28869116` to evaluate stats MLP, XceptionTime, and TemporalAgg on the salvaged checkpoint |
| 28869120 | `summarize_adaptive_eval_metrics.py` | Queued `afterany` on 28869117/28869118/28869119; writes `adaptive_jepa_lora_metrics_after_28869085_1gpu_afterany.{json,csv,md}` |
| 28868419 | `run_item_3_4_fold0_adaptive_capped_temporal_cache_train.sbatch` | Uniformly capped adaptive windows, max 12 and 20, sample-balanced PCA, quick temporal-stats recipes |
| 28868508 | `pd_hand_item_3_4_fold0_adaptive_windowmil_learnedstats_frozen_scorer.yaml` | Freeze the fixed-best per-window CORN scorer, then learn a small mask-aware stats MLP over per-window CORN probabilities/logits |
| 28868563 | `run_item_3_4_fold0_windowmil_calibration_dumps.sbatch` | Completed train-split score dumps plus train-fitted threshold/blend calibration; output `windowmil_trainfit_calibration_28868563.json` |

The window-MIL head is intentionally initialized from the fixed-best CORN checkpoint by keeping the `pooler.*` and `linear.*` parameter names unchanged. It tests whether the fixed model has per-window severity signal that should be pooled with MIL instead of averaging embeddings.

The learned-stats WindowMIL branch is the next aggregation-specific test. It does not fine-tune V-JEPA or the per-window attentive scorer; `encoder_init_path` and `classifier_init_path` load the fixed-best supervised LoRA checkpoint, `freeze_scorer: true` freezes the scorer, and only `stats_mlp` trains on masked summary features of the adaptive window logits/probabilities. This isolates whether the fixed top-k=4 pooling rule is leaving signal on the table.

The capped run changes the failure mode being tested. When `adaptive_max_clips` truncates `ceil(duration_s)`, `ClipDataset` now samples window starts uniformly over the clip instead of taking only the first windows. This keeps the adaptive-window implementation faithful to full-video coverage while bounding the number of windows that can dilute the downstream statistics head.

Smoke check on the longest fold-0 validation clip (`duration_s=33.3`) with `adaptive_max_clips=12` returned starts `[0, 88, ..., 968]`, `adaptive_requested_segments=34`, and `temporal_span_rate=1.0`, confirming the capped windows span the full clip.

The adaptive-window JEPA self-supervised path is still unresolved. The earlier 28867116 attempt is superseded: it launched with the adaptive config path but trained from the stale `handcrop_long` checkpoint and aborted. Smoke job 28868979 completed one adaptive JEPA update from `/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt`. Full job 28868982 then completed epoch 1, wrote a valid `latest.pth.tar`, and failed in epoch 2 with CUDA OOM while processing many adaptive windows from a long clip. To preserve exact adaptive windows instead of capping the training data, `app/vjepa_2_1/train.py` now supports `optimization.clip_microbatch_size`; the resumed job 28869066 uses microbatches of 4 clip windows and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Downstream merge/eval/summary jobs 28869067-28869071 are queued behind it, but the 4-GPU start estimate is slow. A separate one-GPU fallback, 28869085, is seeded by copying the 28868982 epoch-1 `latest.pth.tar` into the `_adaptive_windows_1gpu` folder and has its own afterok merge/eval/summary chain 28869086-28869090. The 1-GPU job is running, resumed cleanly from epoch 1, printed `Using clip_microbatch_size=4`, and reached epoch 2 iteration 30 with GPU memory around 11.4GB. Because a full 25-epoch one-GPU run is too slow, watcher 28869125 will cancel after `Epoch 3` appears, which should mean the epoch-2 checkpoint has already been saved; afterany salvage jobs 28869116-28869120 will then merge/evaluate that latest checkpoint. The epoch-1 snapshot merged successfully, but partial e1 diagnostics were weak: stats MLP 30.3% / QWK 0.000, TemporalAgg 31.8% / QWK 0.036, and XceptionTime 30.3% / QWK 0.000.

## Higher-Water Mark Outside The Attentive Probe

The current fixed attentive-CORN run is not the best V-JEPA-derived result in the workspace. The cached-feature temporal-stats pipeline in `scripts/pd_hand/train_temporal_encoder.py` is stronger:

- Reproduced cross-sweep ensemble: `cross_sweep_ensemble_refresh_20260521.json`
- Candidates loaded: 732 saved val-prob vectors from 16 shard result files
- Best single model: QWK 0.676, acc 48.5
- Best greedy V-JEPA-only ensemble: QWK 0.729, acc 56.1, MAE 0.439
- Best strict subject-OOF-selected V-JEPA-only stacker: QWK 0.689, acc 45.5, MAE 0.591

After mapping the cached-feature NPZ order back to the fold CSV, an OOF-fitted score fusion with kinematic predictions reaches QWK 0.809 / acc 63.6. A validation-tuned fusion improves the fold-0 validation readout to QWK 0.829 / acc 71.2. The former is the honest fold-0 fusion audit; the latter is diagnostic only. Together they show V-JEPA is not redundant with the kinematic model, but the attentive-CORN path is the weak branch and pure V-JEPA still trails hand kinematics.
