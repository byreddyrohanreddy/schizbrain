import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute AUC, F1, Accuracy, Sensitivity, Specificity."""
    
    # ── Youden's J Dynamic Thresholding ──
    try:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = float(thresholds[optimal_idx])
    except:
        pass  # Fallback to 0.5 if ROC curve fails (e.g. single class batch)

    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    tp = ((preds == 1) & (labels == 1)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        "auc":         round(float(auc), 4),
        "f1":          round(float(f1), 4),
        "accuracy":    round(float(acc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
    }


# ==============================================================
# MIXUP AUGMENTATION
# Interpolates between scan pairs — forces smooth decision
# boundaries, very effective for small medical datasets.
# ==============================================================

def mixup_batch(
    volumes: torch.Tensor,
    labels: torch.Tensor,
    ages: torch.Tensor,
    alpha: float = 0.4,
):
    """
    Apply MixUp to a batch of 3D MRI scans.

    lam ~ Beta(alpha, alpha). Mixed sample:
        x_mix = lam * x_i + (1-lam) * x_j
        y_mix = lam * y_i + (1-lam) * y_j

    Ages are also interpolated (continuous feature).
    Genders are NOT mixed (categorical — kept from primary sample).
    alpha=0.4 is strong (recommended for medical imaging).
    alpha=0.0 disables MixUp (pass-through).
    """
    if alpha <= 0 or volumes.size(0) < 2:
        return volumes, labels, ages

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # primary sample always dominates

    idx = torch.randperm(volumes.size(0), device=volumes.device)
    mixed_volumes = lam * volumes + (1 - lam) * volumes[idx]
    mixed_labels  = lam * labels  + (1 - lam) * labels[idx]
    mixed_ages    = lam * ages    + (1 - lam) * ages[idx]

    return mixed_volumes, mixed_labels, mixed_ages


# ==============================================================
# TRAIN EPOCH V5 — MixUp, no DANN
# ==============================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    accumulation_steps: int = 4,
    mixup_alpha: float = 0.4,
) -> Tuple[float, float]:
    """
    Train one epoch with MixUp augmentation. No DANN.
    DataLoader yields: (volume, label, age, gender, site)
    Model receives:    model(volume, age, gender)
    """
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc="  Training", leave=False,
                dynamic_ncols=True, position=0)

    optimizer.zero_grad(set_to_none=True)

    for i, (volumes, labels, ages, genders, _sites) in enumerate(pbar):

        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        ages    = ages.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)

        # ── MixUp ─────────────────────────────────────────────
        volumes, labels, ages = mixup_batch(
            volumes, labels, ages, alpha=mixup_alpha
        )

        with amp.autocast():
            logits = model(volumes, ages, genders)
            loss   = criterion(logits, labels.float())
            probs  = torch.sigmoid(torch.clamp(logits.float(), min=-20, max=20))

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += (loss.item() * accumulation_steps)
        all_labels.extend(labels.cpu().numpy().flatten())
        all_probs.extend(probs.detach().cpu().numpy().flatten())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return avg_loss, auc


# ==============================================================
# VALIDATE EPOCH V3
# Now passes age and gender to model
# ==============================================================

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate one epoch — no MixUp, no DANN."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for volumes, labels, ages, genders, _sites in tqdm(
        loader, desc="  Validating", leave=False,
        dynamic_ncols=True, position=0
    ):
        volumes = volumes.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        ages    = ages.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)

        with amp.autocast():
            logits = model(volumes, ages, genders)
            loss   = criterion(logits, labels.float())
            probs  = torch.sigmoid(torch.clamp(logits.float(), min=-20, max=20))

        total_loss += loss.item()
        batch_labels = labels.cpu().numpy().flatten()
        batch_probs  = probs.cpu().numpy().flatten()
        # Guard against NaN (can happen if model weights corrupted by gradient explosion)
        if not np.isnan(batch_probs).any():
            all_labels.extend(batch_labels)
            all_probs.extend(batch_probs)

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics


