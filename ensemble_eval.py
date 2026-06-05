"""
SchizoBrain Ensemble Evaluator
===============================
Loads all 5 fold checkpoints and averages their predictions for a free AUC boost.
Ensemble typically adds +0.02-0.04 AUC at zero training cost.

Usage:
    python ensemble_eval.py

This script:
1. Loads metadata CSV
2. For each fold: loads that fold's checkpoint, runs inference on its val set
3. Also runs ALL 5 models on EVERY sample and averages probabilities
4. Reports per-fold AUC and ensemble AUC
"""

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from hybrid_model_v2 import SchizoBrain, AgeNormalizer
from Trainer import MRIDataset

CHECKPOINT_DIR = "experiments/checkpoints"
METADATA_PATH  = "data/metadata_pt.csv"
SEED           = 42
NUM_FOLDS      = 5
BATCH_SIZE     = 4
NUM_WORKERS    = 2

MODEL_CONFIG = dict(
    pretrained=False,   # weights come from checkpoint, no hub load needed
    frozen=False,
    embed_dim=256,
    num_heads=4,
    num_layers=6,       # V4: upgraded from 4
    mlp_ratio=2,
    attn_dropout=0.2,
    ffn_dropout=0.2,
    head_dropout=0.3,   # V4: upgraded from 0.2
    clinical_embed_dim=32,
    hidden_dim=128,
)

TTA_N_AUGS = 5  # original + 4 deterministic flips


def apply_tta_aug(volumes: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """
    Deterministic TTA augmentations for 3D MRI volumes.
    Shape: (B, 1, D, H, W)

    aug_idx=0 : original (no transform)
    aug_idx=1 : flip D (axial)
    aug_idx=2 : flip H (coronal)
    aug_idx=3 : flip W (sagittal)
    aug_idx=4 : flip D+H
    """
    if aug_idx == 0:
        return volumes
    elif aug_idx == 1:
        return volumes.flip(2)
    elif aug_idx == 2:
        return volumes.flip(3)
    elif aug_idx == 3:
        return volumes.flip(4)
    elif aug_idx == 4:
        return volumes.flip(2).flip(3)
    return volumes


def load_metadata(path):
    rows = list(csv.DictReader(open(path)))
    filepaths = [r["filepath"] for r in rows]
    labels    = [int(r["label"]) for r in rows]
    ages      = [float(r["age"]) for r in rows]
    genders   = [r["gender"] for r in rows]
    sites     = [int(r["site"]) for r in rows]
    return filepaths, labels, ages, genders, sites


def load_model(checkpoint_path, device):
    model = SchizoBrain(**MODEL_CONFIG).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def infer(model, loader, device, use_tta=True):
    """
    Run inference with optional Test Time Augmentation.

    TTA: runs each scan through TTA_N_AUGS augmentations
    (original + 4 flips) and averages probabilities.
    Typically adds +0.02-0.03 AUC at zero training cost.
    """
    all_probs, all_labels = [], []
    n_augs = TTA_N_AUGS if use_tta else 1

    with amp.autocast():
        for volumes, labels, ages, genders, sites in loader:
            volumes = volumes.to(device, non_blocking=True)
            ages    = ages.to(device, non_blocking=True)
            genders = genders.to(device, non_blocking=True)

            # Run all TTA augmentations and stack probabilities
            aug_probs = []
            for aug_idx in range(n_augs):
                aug_vol = apply_tta_aug(volumes, aug_idx)
                logits  = model(aug_vol, ages, genders)
                probs   = torch.sigmoid(logits.float()).cpu().numpy().flatten()
                aug_probs.append(probs)

            # Average across augmentations
            mean_probs = np.mean(aug_probs, axis=0)
            all_probs.extend(mean_probs)
            all_labels.extend(labels.numpy().flatten())

    return np.array(all_labels), np.array(all_probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    filepaths, labels, ages, genders, sites = load_metadata(METADATA_PATH)
    labels_array = np.array(labels)
    indices      = np.arange(len(labels))

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(indices, labels_array))

    print(f"\n{'='*60}")
    print("SchizoBrain Ensemble Evaluation")
    print(f"{'='*60}")

    # ── Per-fold evaluation (each model on its own val set) ──────
    print("\n[1] Per-Fold Checkpoint Evaluation")
    fold_aucs = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold = fold_idx + 1
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold}.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold}: checkpoint not found — skipping")
            continue

        # Fit age normalizer on training fold only
        train_ages = [ages[i] for i in train_idx]
        age_norm = AgeNormalizer()
        age_norm.fit(train_ages)

        import pandas as pd
        val_df = pd.DataFrame({
            "filepath": [filepaths[i] for i in val_idx],
            "label":    [labels[i] for i in val_idx],
            "age":      [ages[i] for i in val_idx],
            "gender":   [genders[i] for i in val_idx],
            "site":     [sites[i] for i in val_idx],
        })
        val_dataset = MRIDataset(val_df, age_normalizer=age_norm, transform=None)
        val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)

        model = load_model(ckpt_path, device)
        true_labels, probs = infer(model, val_loader, device)

        auc = roc_auc_score(true_labels, probs)
        fold_aucs.append(auc)
        preds = (probs >= 0.5).astype(int)
        f1  = f1_score(true_labels, preds, zero_division=0)
        acc = accuracy_score(true_labels, preds)
        print(f"  Fold {fold}: AUC={auc:.4f} | F1={f1:.4f} | Acc={acc:.4f}  ({len(val_idx)} samples)")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if fold_aucs:
        print(f"\n  Individual mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # ── Full ensemble: average all 5 models on ALL data ──────────
    print("\n[2] Full Ensemble (all 5 models on all data, averaged per fold's val set)")

    # Load all 5 models
    models = {}
    for fold in range(1, NUM_FOLDS + 1):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(ckpt_path):
            models[fold] = load_model(ckpt_path, device)
            print(f"  Loaded fold {fold} checkpoint")

    if len(models) < 2:
        print("  Need at least 2 checkpoints for ensemble — aborting")
        return

    ensemble_labels, ensemble_probs_sum, ensemble_counts = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold = fold_idx + 1

        train_ages = [ages[i] for i in train_idx]
        age_norm = AgeNormalizer()
        age_norm.fit(train_ages)

        import pandas as pd
        val_df = pd.DataFrame({
            "filepath": [filepaths[i] for i in val_idx],
            "label":    [labels[i] for i in val_idx],
            "age":      [ages[i] for i in val_idx],
            "gender":   [genders[i] for i in val_idx],
            "site":     [sites[i] for i in val_idx],
        })
        val_dataset = MRIDataset(val_df, age_normalizer=age_norm, transform=None)
        val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=NUM_WORKERS,
                                 pin_memory=True)

        # Average predictions from ALL models on this val set
        fold_probs = []
        for m_fold, model in models.items():
            true_labels, probs = infer(model, val_loader, device)
            fold_probs.append(probs)

        ensemble_pred = np.mean(fold_probs, axis=0)
        auc = roc_auc_score(true_labels, ensemble_pred)
        preds = (ensemble_pred >= 0.5).astype(int)
        f1  = f1_score(true_labels, preds, zero_division=0)
        acc = accuracy_score(true_labels, preds)
        print(f"  Fold {fold} ensemble AUC={auc:.4f} | F1={f1:.4f} | Acc={acc:.4f}")
        ensemble_labels.append((true_labels, ensemble_pred))

    # Global ensemble AUC (all folds combined)
    all_true = np.concatenate([x[0] for x in ensemble_labels])
    all_pred = np.concatenate([x[1] for x in ensemble_labels])
    global_auc = roc_auc_score(all_true, all_pred)
    global_f1  = f1_score(all_true, (all_pred >= 0.5).astype(int), zero_division=0)
    global_acc = accuracy_score(all_true, (all_pred >= 0.5).astype(int))

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE RESULT (all {len(models)} models, all {len(all_true)} samples)")
    print(f"  AUC:      {global_auc:.4f}")
    print(f"  F1:       {global_f1:.4f}")
    print(f"  Accuracy: {global_acc:.4f}")
    print(f"{'='*60}")

    if fold_aucs:
        improvement = global_auc - np.mean(fold_aucs)
        print(f"\n  Ensemble vs individual mean: {improvement:+.4f} AUC")


if __name__ == "__main__":
    main()
