# May 24 reproduction guide

This guide reproduces the May 24 item 3.4 result:

`kin_only 0.3615 -> kin_only + V-JEPA OOF 0.4090`, delta `+0.0475`.

Run from:

```bash
cd /gpfs/milgram/pi/scherzer/yl2428/vjepa2
```

The data root used by the scripts is:

```bash
/gpfs/milgram/pi/scherzer/yl2428/pd-analysis/outputs/foundation_model_minimal_hand_tasks_may19
```

## 1. Cache V-JEPA sliding-window features

This step reads the item 3.4 fold CSVs and LoRA-merged checkpoints, then writes
train/val NPZ feature caches for folds 0-4.

```bash
sbatch scripts/pd_hand/run_may24_cache_vjepa_features.sbatch
```

The cache script supports fixed or adaptive sliding windows. The May 24 runner
uses adaptive coverage:

```bash
python scripts/pd_hand/cache_vjepa21_temporal_embeddings.py \
  --config configs/eval_2_1/pd_hand_may24_vitl384_lora_cache.yaml \
  --dataset-train <fold_train.csv> \
  --dataset-val <fold_val.csv> \
  --checkpoint <merged_lora_checkpoint.pt> \
  --out-dir <cache_out_dir> \
  --adaptive-num-clips \
  --adaptive-min-clips 4 \
  --adaptive-max-clips 48
```

Expected cache location:

```bash
${ROOT}/vjepa2_embeddings/item_3_4/fold_${FOLD}_loraft_aggressive_latest_adaptive
```

## 2. Generate leak-free V-JEPA OOF probabilities

```bash
sbatch scripts/pd_hand/run_may24_oof_vjepa_probs.sbatch
```

This uses `oof_vjepa_combiner_predictions.py` with `--no-val-tuning`, so the
held-out validation fold is not used for epoch/model selection. Inner splits are
subject-disjoint and the script asserts that train/holdout subjects do not
overlap.

Expected OOF output:

```bash
${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_loraft_latest_clean.npz
```

## 3. Join OOF probabilities onto kinematic CSVs

```bash
sbatch scripts/pd_hand/run_may24_augment_oof_features.sbatch
```

The join is by `clip_path` through cache sample indices. It fails on duplicate
clips, invalid sample indices, or missing OOF probabilities; it does not zero
fill missing rows.

Expected joined CSVs:

```bash
${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/train_features_with_oof.csv
${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/val_features_with_oof.csv
```

## 4. Run the fair multiseed stacker

```bash
sbatch scripts/pd_hand/run_may24_fair_multiseed.sbatch
```

The stacker evaluates five feature sets:

- `kin_only`
- `kin_only_plus_vjepa_all`
- `kin_plus_dx` (diagnosis one-hots plus context columns; reference only)
- `kin_plus_dx_plus_vjepa_all`
- `vjepa_only`

For the headline result, only compare `kin_only` against
`kin_only_plus_vjepa_all`.

The fair `kin_only` set excludes diagnosis, diagnosis one-hots, all `item35*`
columns, manifest metadata, IDs, and labels. The script raises an error if any
item 3.5 column enters an evaluated feature set.

The secondary `kin_plus_dx` reference keeps the historical May 24 context
surface for compatibility. It is not the headline fair baseline.

Expected per-fold output:

```bash
${ROOT}/hybrid/item_3_4/fold_${FOLD}/oof_stack/multiseed_oof_probs_kinonly.json
```

Historical May 24 artifacts used
`multiseed_oof_probs_kinonly_clean.json` for fold 0. The aggregator supports
both names.

## 5. Aggregate the 5 folds

```bash
/gpfs/milgram/apps/avx2/software/miniconda/24.11.3/bin/conda run \
  --no-capture-output -n video-llama \
  python scripts/pd_hand/aggregate_kinonly_5fold.py
```

Expected headline output:

```text
fair_kin_only_plus_vjepa_oof
mean: base +0.3615  hybrid +0.4090  delta +0.0475  positive folds 4/5
pooled seed-fold: wins 35/50  p=6.227e-05; fold-mean p=0.09375
```

Aggregate JSON:

```bash
${ROOT}/r2_kin_scan/kinonly_5fold_aggregate.json
```

## Notes for reruns

- Use the `video-llama` conda environment.
- Do not use validation-tuned combiner selection for the May 24 claim.
- Do not add `dx`, `dx_*`, or `item35*` to the headline fair baseline.
- Treat `kin_plus_dx` as a reference clinical/context setting, not the raw-video claim.
- If reproducing from historical artifacts only, start at step 5.
