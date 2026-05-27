#!/usr/bin/env python
"""Subject-disjoint OOF V-JEPA combiner predictions.

Splits an outer-fold train set by subject into inner folds, trains one temporal
CORN combiner per inner fold, and stitches inner-validation predictions into
train-set OOF probabilities aligned to the V-JEPA cache row order.

Also produces outer-validation predictions using a combiner trained on the full
outer-fold train set.  Use --no-val-tuning for the May 24 clean protocol: it
returns final-epoch probabilities instead of best-by-validation-QWK snapshots.

Output: NPZ with {train_probs_cache_order[N=404, 5], val_probs_cache_order[N=107, 5],
                  train_cache_sample_indices, val_cache_sample_indices}.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_corn_combiner import (  # type: ignore
    NUM_CLASSES, TemporalCornCombiner, corn_loss, corn_predict,
    corn_pos_weight_from_labels, cosine_warmup,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-train", required=True)
    p.add_argument("--cache-val", required=True)
    p.add_argument("--fold-train-csv", required=True)
    p.add_argument("--fold-val-csv", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--num-inner-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--num-segments-for-pos", type=int, default=48)
    p.add_argument("--tubelets-per-segment", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-val-tuning", action="store_true",
                   help="Disable early stopping on outer val. Train for exactly --epochs and use the final-epoch predictions.")
    return p.parse_args()


def load_cache(path: str | Path):
    p = Path(path)
    d = np.load(p, allow_pickle=False)
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.int64)
    mask = d["mask"].astype(bool) if "mask" in d.files else np.ones(x.shape[:2], dtype=bool)
    cov = [json.loads(str(s)) for s in d["coverage"]] if "coverage" in d.files else None
    if cov is None:
        raise ValueError(f"{p} is missing coverage rows; cannot map cache rows to subject IDs")
    return x, y, mask, cov


def cache_to_subject_ids(coverage, fold_csv_path):
    """For each cache row, return the subject_id (via sample_index → fold CSV row)."""
    fold_rows = list(csv.DictReader(open(fold_csv_path)))
    out = []
    for c in coverage:
        si = int(round(float(c.get("sample_index", -1))))
        if 0 <= si < len(fold_rows):
            out.append(fold_rows[si].get("subject_id", "UNK"))
        else:
            out.append("UNK")
    if any(subject == "UNK" for subject in out):
        raise ValueError(f"Could not map all cache rows to subject IDs via {fold_csv_path}")
    return np.asarray(out, dtype=object)


def cache_sample_indices(coverage) -> np.ndarray:
    return np.asarray([int(round(float(c.get("sample_index", -1)))) for c in coverage], dtype=np.int64)


def train_combiner_and_predict(
    x_tr, mask_tr, y_tr, x_va, mask_va, y_va, args, scales_meta,
    pos_weight=None, device="cuda:0", select_on_val=True,
):
    """Train a combiner and return probabilities on x_va.

    With select_on_val=False, this returns final-epoch probabilities and never
    selects the best validation epoch.  That is the clean May 24 setting.
    """
    # Build a ScaleCache-like wrapper (only used for input_dim and pos info)
    class _Scale:
        def __init__(self, x, num_segments_for_pos, tubelets_per_segment):
            self.x_train = x
            self.num_segments_for_pos = num_segments_for_pos
            self.tubelets_per_segment = tubelets_per_segment
    scale_proxy = _Scale(x_tr, args.num_segments_for_pos, args.tubelets_per_segment)

    model = TemporalCornCombiner(
        scales=[scale_proxy], d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_mult=4, dropout=0.3, num_classes=NUM_CLASSES,
        aux_dim=0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    best_qwk = -2.0
    best_state = None
    epochs_since = 0
    rng = np.random.default_rng(args.seed)

    n_train = len(y_tr)
    for epoch in range(args.epochs):
        lr = cosine_warmup(epoch, 8, 100, 3e-4, 3e-5)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        order = rng.permutation(n_train)
        for i in range(0, n_train, 16):
            idx = order[i:i + 16]
            xs = torch.from_numpy(x_tr[idx]).to(device, dtype=torch.float32)
            ms = torch.from_numpy(mask_tr[idx]).to(device, dtype=torch.bool)
            ys = torch.from_numpy(y_tr[idx]).to(device, dtype=torch.long)
            out = model([(xs, ms)])
            logits = out[0] if isinstance(out, tuple) else out
            loss = corn_loss(logits, ys, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Evaluate on val
        model.eval()
        with torch.no_grad():
            preds_chunks, probs_chunks = [], []
            for i in range(0, len(y_va), 16):
                xs = torch.from_numpy(x_va[i:i+16]).to(device, dtype=torch.float32)
                ms = torch.from_numpy(mask_va[i:i+16]).to(device, dtype=torch.bool)
                out = model([(xs, ms)])
                logits = out[0] if isinstance(out, tuple) else out
                p_cond = torch.sigmoid(logits)
                p_cum = torch.cumprod(p_cond, dim=1)
                B = logits.shape[0]
                p_class = torch.zeros(B, NUM_CLASSES, device=logits.device)
                p_class[:, 0] = 1.0 - p_cum[:, 0]
                for k in range(1, NUM_CLASSES - 1):
                    p_class[:, k] = p_cum[:, k - 1] - p_cum[:, k]
                p_class[:, NUM_CLASSES - 1] = p_cum[:, NUM_CLASSES - 2]
                p_class = p_class.clamp(min=0).softmax(dim=1) if False else p_class
                preds_chunks.append(corn_predict(logits).cpu().numpy())
                probs_chunks.append(p_class.cpu().numpy())
            yhat = np.concatenate(preds_chunks)
            probs = np.concatenate(probs_chunks)
        from sklearn.metrics import cohen_kappa_score
        qwk = cohen_kappa_score(y_va, yhat, weights="quadratic")
        # Always cache the latest probs (used when select_on_val=False)
        last_state = {"probs": probs.copy(), "qwk": float(qwk), "epoch": epoch}
        if select_on_val:
            if qwk > best_qwk:
                best_qwk = qwk
                best_state = {"probs": probs.copy(), "qwk": float(qwk), "epoch": epoch}
                epochs_since = 0
            else:
                epochs_since += 1
                if epochs_since >= args.patience:
                    break
    if select_on_val:
        if best_state is None:
            raise RuntimeError("No best validation state was recorded")
        return best_state
    return last_state


def assert_subject_disjoint(subjects: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, label: str) -> None:
    train_subjects = set(subjects[train_idx])
    val_subjects = set(subjects[val_idx])
    overlap = train_subjects.intersection(val_subjects)
    if overlap:
        sample = sorted(str(x) for x in overlap)[:8]
        raise ValueError(f"{label}: subject overlap between train and val: {sample}")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_tr, y_tr, mask_tr, cov_tr = load_cache(args.cache_train)
    x_va, y_va, mask_va, cov_va = load_cache(args.cache_val)
    print(f"train cache: {x_tr.shape}, val cache: {x_va.shape}")

    # Subject IDs in cache row order
    subj_tr = cache_to_subject_ids(cov_tr, args.fold_train_csv)
    n_subj = len(set(subj_tr))
    print(f"n unique subjects in train cache: {n_subj}")
    # Stratify by label (median per subject); group by subject id
    sgk = StratifiedGroupKFold(n_splits=args.num_inner_folds, shuffle=True, random_state=args.seed)
    train_probs = np.zeros((len(y_tr), NUM_CLASSES), dtype=np.float32)
    for k_fold, (inner_tr_idx, inner_va_idx) in enumerate(sgk.split(x_tr, y_tr, groups=subj_tr)):
        assert_subject_disjoint(subj_tr, inner_tr_idx, inner_va_idx, f"inner fold {k_fold + 1}")
        print(f"\n=== inner fold {k_fold+1}/{args.num_inner_folds}: "
              f"inner_tr={len(inner_tr_idx)}, inner_va={len(inner_va_idx)} ===")
        pos_w = corn_pos_weight_from_labels(y_tr[inner_tr_idx]).to(device)
        result = train_combiner_and_predict(
            x_tr[inner_tr_idx], mask_tr[inner_tr_idx], y_tr[inner_tr_idx],
            x_tr[inner_va_idx], mask_tr[inner_va_idx], y_tr[inner_va_idx],
            args, scales_meta=None, pos_weight=pos_w, device=device,
            select_on_val=not args.no_val_tuning,
        )
        which = "final-epoch" if args.no_val_tuning else "best-epoch"
        print(f"  inner-fold {k_fold+1} {which} inner-val QWK: {result['qwk']:.4f}  ep={result['epoch']}")
        train_probs[inner_va_idx] = result["probs"]

    # Train on full fold-0 train, predict on fold-0 val
    print("\n=== full fold-0 train → val predictions ===")
    pos_w = corn_pos_weight_from_labels(y_tr).to(device)
    val_result = train_combiner_and_predict(
        x_tr, mask_tr, y_tr, x_va, mask_va, y_va,
        args, scales_meta=None, pos_weight=pos_w, device=device,
        select_on_val=not args.no_val_tuning,
    )
    val_probs = val_result["probs"]
    which = "final-epoch" if args.no_val_tuning else "best-epoch"
    print(f"  full-train fold-0 val ({which}) QWK: {val_result['qwk']:.4f}  ep={val_result['epoch']}")

    train_csv_idx = cache_sample_indices(cov_tr)
    val_csv_idx = cache_sample_indices(cov_va)
    np.savez_compressed(
        args.out_npz,
        train_probs_cache_order=train_probs.astype(np.float32),
        val_probs_cache_order=val_probs.astype(np.float32),
        train_cache_sample_indices=train_csv_idx,
        val_cache_sample_indices=val_csv_idx,
        train_y=y_tr.astype(np.int64),
        val_y=y_va.astype(np.int64),
        config=json.dumps(vars(args), sort_keys=True),
    )
    print(f"\nwrote {args.out_npz}")


if __name__ == "__main__":
    main()
