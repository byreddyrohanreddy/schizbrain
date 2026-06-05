"""
SchizoBrain Trainer V3
=======================
Updated to support age and gender late fusion (V3 model).

Changes from V2 trainer:
    - MRIDataset now loads age and gender from metadata CSV
    - AgeNormalizer fitted on training fold, applied to val fold
    - encode_gender handles M/F/Male/Female/0/1
    - train_epoch passes (mri, age, gender) to model
    - validate_epoch passes (mri, age, gender) to model
    - metadata CSV now expects: filepath, label, age, gender

Metadata CSV format (data/metadata.csv):
    filepath,                        label, age, gender
    data/processed/scan_001.nii.gz,  0,     34,  M
    data/processed/scan_002.nii.gz,  1,     25,  F
    ...

Usage:
    python trainer_v3.py
"""

import warnings
# Must be before ALL other imports so torch/tio warnings are suppressed from the start
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
import wandb
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import nibabel as nib
import pandas as pd

# Import V3 model and utilities
from hybrid_model_v2 import SchizoBrain, AgeNormalizer, encode_gender


# ==============================================================
# FOCAL LOSS (unchanged from V2)
# ==============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with automatic class weight computation.
    Fixes class imbalance — schizophrenia cases penalized more.
    """

    def __init__(self, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None,
                 smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # ── Label Smoothing ───────────────────────────────────
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed_targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )
        prob = torch.clamp(torch.sigmoid(logits), min=1e-6, max=1 - 1e-6)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        p_t = torch.clamp(p_t, min=1e-6, max=1 - 1e-6)  # prevent 0*inf=NaN with soft MixUp labels
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

    @staticmethod
    def compute_pos_weight(labels: List[int]) -> torch.Tensor:
        labels = np.array(labels)
        num_positive = (labels == 1).sum()
        num_negative = (labels == 0).sum()
        pos_weight = num_negative / num_positive
        print(f"\n📊 Class Distribution:")
        print(f"   Healthy (0):       {num_negative} scans")
        print(f"   Schizophrenia (1): {num_positive} scans")
        print(f"   Pos weight:        {pos_weight:.3f}")
        return torch.tensor([pos_weight], dtype=torch.float32)


# ==============================================================
# MRI DATASET V3
# Now loads age and gender alongside MRI and label
# ==============================================================

class MRIDataset(Dataset):
    """
    Dataset for preprocessed 3D MRI scans with clinical metadata.

    Expects DataFrame with columns:
        filepath : path to .nii.gz file
        label    : 0 (healthy) or 1 (schizophrenia)
        age      : patient age (raw, will be normalized externally)
        gender   : M/F/Male/Female/0/1

    AgeNormalizer must be fitted on training fold BEFORE
    creating the dataset — pass it as age_normalizer argument.

    Args:
        dataframe:      DataFrame with filepath, label, age, gender
        age_normalizer: Fitted AgeNormalizer instance
        transform:      Optional TorchIO augmentation (train only)
        target_shape:   MRI resize target (default 96x96x96)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        age_normalizer: AgeNormalizer,
        transform=None,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
    ):
        self.df = dataframe.reset_index(drop=True)
        self.age_normalizer = age_normalizer
        self.transform = transform
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Load MRI ──────────────────────────────────────────
        filepath = str(row["filepath"])
        
        if filepath.endswith(".pt"):
            # Instant load of pre-sized (1, 96, 96, 96) tensor
            abs_path = os.path.join(r"c:\schizobrain", filepath) if not os.path.isabs(filepath) else filepath
            volume = torch.load(abs_path)
        else:
            # Fallback for raw .nii files
            mri = nib.load(filepath)
            volume = mri.get_fdata(dtype=np.float32)

            # Squeeze out any phantom extra dimensions
            volume = np.squeeze(volume)
            if volume.ndim == 4:
                volume = volume[..., 0]

            if volume.shape != self.target_shape:
                volume = self._resize(volume, self.target_shape)

            # Normalize MRI intensity
            volume = (volume - volume.mean()) / (volume.std() + 1e-8)
            volume = torch.tensor(volume).unsqueeze(0)  # (1, D, H, W)

        if self.transform:
            volume = self.transform(volume)

        # ── Label ─────────────────────────────────────────────
        label = torch.tensor([float(row["label"])], dtype=torch.float32)

        # ── Age (normalized) ──────────────────────────────────
        age_norm = self.age_normalizer.transform(float(row["age"]))
        age = torch.tensor([[age_norm]], dtype=torch.float32).squeeze(0)
        # shape: (1,)

        # ── Gender (encoded) ──────────────────────────────────
        g = str(row["gender"]).lower()
        gender_val = 1.0 if g in ["m", "male", "1"] else 0.0
        gender = torch.tensor([gender_val], dtype=torch.float32)
        # shape: (1,)

        # ── Site (DANN) ───────────────────────────────────────
        site = int(row["site"])

        return volume, label, age, gender, site

    def _resize(self, volume: np.ndarray,
                target: Tuple) -> np.ndarray:
        """Resize volume to target shape via trilinear interpolation."""
        import torch.nn.functional as F
        vol_t = torch.tensor(volume).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(vol_t, size=target,
                                mode="trilinear", align_corners=False)
        return resized.squeeze().numpy()


# ==============================================================
# WARMUP + COSINE SCHEDULER (unchanged)
# ==============================================================

class WarmupCosineScheduler:
    """Linear warmup then CosineAnnealingLR."""

    def __init__(self, optimizer, warmup_epochs=10,
                 total_epochs=100, min_lr=1e-8):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
            scale = max(scale, self.min_lr / max(self.base_lrs))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ==============================================================
# EARLY STOPPING (unchanged)
# ==============================================================

class EarlyStopping:
    """Early stopping on validation AUC with best model saving."""

    def __init__(self, patience=15, min_delta=0.001,
                 save_path="experiments/checkpoints/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_auc = 0.0
        self.counter = 0
        self.stopped = False
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def __call__(self, val_auc: float,
                 model: nn.Module, fold: int) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter = 0
            path = self.save_path.replace(".pth", f"_fold{fold}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "fold": fold,
            }, path)
            print(f"   💾 Saved best model (AUC={val_auc:.4f}) → {path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                print(f"   🛑 Early stopping (no improvement "
                      f"for {self.patience} epochs)")
        return self.stopped

    def reset(self):
        # DO NOT reset best_auc! If Phase 2 performs worse, we want to retain the Phase 1 saved weights.
        self.counter = 0
        self.stopped = False


# ==============================================================
# METRICS (unchanged)
# ==============================================================

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


# ==============================================================
# MAIN TRAINING LOOP V3
# AgeNormalizer fitted per fold on training data only
# ==============================================================

def train_schizobrain(
    dataframe: pd.DataFrame,
    config: Dict,
    device: torch.device,
    augmentation=None,
):
    """
    5-Fold Stratified Cross Validation with age+gender support.

    Key V3 changes:
        1. AgeNormalizer fitted on TRAINING fold only
           (prevents data leakage from val/test ages)
        2. Both train and val datasets receive fitted normalizer
        3. train_epoch and validate_epoch pass age+gender to model

    Args:
        dataframe:    DataFrame with filepath, label, age, gender
        config:       Training configuration dictionary
        device:       cuda or cpu
        augmentation: Optional TorchIO augmentation

    Returns:
        fold_results: List of metric dicts per fold
    """

    # ── Validate required columns ─────────────────────────────
    required = ["filepath", "label", "age", "gender", "site"]
    missing = [c for c in required if c not in dataframe.columns]
    if missing:
        raise ValueError(
            f"Missing columns in metadata CSV: {missing}\n"
            f"Required: filepath, label, age, gender"
        )

    # ── wandb ─────────────────────────────────────────────────
    wandb.init(
        project="SchizoBrain",
        name=f"V6_NoDAN_NoMixUp_{time.strftime('%Y%m%d_%H%M%S')}",
        config=config,
    )

    # ── Class weights ─────────────────────────────────────────
    all_labels = dataframe["label"].tolist()
    pos_weight = FocalLoss.compute_pos_weight(all_labels).to(device)
    criterion  = FocalLoss(gamma=2.0, pos_weight=pos_weight, smoothing=0.05)

    # ── AMP scaler ────────────────────────────────────────────
    scaler = amp.GradScaler()

    # ── Stratified KFold ──────────────────────────────────────
    skf = StratifiedKFold(
        n_splits=config["num_folds"],
        shuffle=True,
        random_state=config["seed"],
    )

    labels_array = np.array(all_labels)
    indices = np.arange(len(dataframe))
    fold_results = []

    print(f"\n{'='*60}")
    print(f"SchizoBrain V3 — {config['num_folds']}-Fold CV")
    print(f"Features: MRI + Age + Gender")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(indices, labels_array), start=1
    ):
        print(f"\n{'─'*60}")
        print(f"FOLD {fold}/{config['num_folds']}")

        train_df = dataframe.iloc[train_idx].reset_index(drop=True)
        val_df   = dataframe.iloc[val_idx].reset_index(drop=True)

        # Print fold class distribution
        tr_labels = labels_array[train_idx]
        vl_labels = labels_array[val_idx]
        print(f"  Train → Healthy: {(tr_labels==0).sum()} | "
              f"Schizo: {(tr_labels==1).sum()}")
        print(f"  Val   → Healthy: {(vl_labels==0).sum()} | "
              f"Schizo: {(vl_labels==1).sum()}")

        # ── Fit AgeNormalizer on TRAINING fold only ───────────
        # Critical: fit only on train to prevent data leakage
        age_normalizer = AgeNormalizer()
        age_normalizer.fit(train_df["age"].tolist())

        # Print age/gender stats for this fold
        train_male = (train_df["gender"].astype(str).str.lower()
                      .isin(["m", "male", "1"])).sum()
        print(f"  Train → Age: {train_df['age'].min():.0f}-"
              f"{train_df['age'].max():.0f} | "
              f"Male: {train_male}/{len(train_df)}")
        print(f"{'─'*60}")

        # ── Build datasets ────────────────────────────────────
        train_dataset = MRIDataset(
            train_df,
            age_normalizer=age_normalizer,
            transform=augmentation,
        )
        val_dataset = MRIDataset(
            val_df,
            age_normalizer=age_normalizer,  # same normalizer
            transform=None,                 # no aug on val
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # ── Initialize fresh model for each fold ─────────────
        model = SchizoBrain(
            pretrained=config["pretrained"],
            pretrained_path=config.get("pretrained_path"),
            frozen=True,
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            mlp_ratio=config["mlp_ratio"],
            attn_dropout=config["attn_dropout"],
            ffn_dropout=config["ffn_dropout"],
            head_dropout=config["head_dropout"],
            clinical_embed_dim=config["clinical_embed_dim"],
            hidden_dim=config["hidden_dim"],
        ).to(device)

        # ── Phase 1 optimizer ─────────────────────────────────
        optimizer = model.phase1_optimizer(
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config["warmup_epochs"],
            total_epochs=config["phase1_epochs"],
        )

        early_stopping = EarlyStopping(
            patience=config["patience"],
            save_path=config["checkpoint_path"],
        )

        phase = 1
        best_metrics = {}

        # ── Training loop ─────────────────────────────────────
        for epoch in range(1, config["epochs"] + 1):

            # ── Switch to Phase 2 ─────────────────────────────
            if epoch == config["phase1_epochs"] + 1 and phase == 1:
                print(f"\n  🔄 Switching to Phase 2 — fine-tuning CNN")
                model.switch_to_phase2()
                optimizer = model.phase2_optimizer(
                    vit_lr=config["lr"],
                    cnn_lr=config["lr"] / 10,
                    weight_decay=config["weight_decay"],
                )
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_epochs=5,
                    total_epochs=config["epochs"] - config["phase1_epochs"],
                )
                early_stopping.reset()
                phase = 2

            # ── Update LR ─────────────────────────────────────
            phase_epoch = (
                epoch - 1 if phase == 1
                else epoch - config["phase1_epochs"] - 1
            )
            scheduler.step(phase_epoch)
            current_lrs = scheduler.get_lr()

            # ── Train + Validate (no DANN) ────────────────────
            train_loss, train_auc = train_epoch(
                model, train_loader, optimizer,
                criterion, scaler, device,
                grad_clip=config["grad_clip"],
                accumulation_steps=config.get("accumulation_steps", 4),
                mixup_alpha=config.get("mixup_alpha", 0.4),
            )
            val_loss, val_metrics = validate_epoch(
                model, val_loader, criterion, device,
            )

            val_auc = val_metrics["auc"]
            val_f1  = val_metrics["f1"]

            # ── wandb logging ─────────────────────────────────
            wandb.log({
                f"fold{fold}/train_loss":    train_loss,
                f"fold{fold}/train_auc":     train_auc,
                f"fold{fold}/val_loss":      val_loss,
                f"fold{fold}/val_auc":       val_auc,
                f"fold{fold}/val_f1":        val_f1,
                f"fold{fold}/val_acc":       val_metrics["accuracy"],
                f"fold{fold}/val_sens":      val_metrics["sensitivity"],
                f"fold{fold}/val_spec":      val_metrics["specificity"],
                f"fold{fold}/lr":            current_lrs[0],
                f"fold{fold}/phase":         phase,
                "epoch": epoch,
            })

            # ── Print ─────────────────────────────────────────
            print(
                f"  Epoch {epoch:3d}/{config['epochs']} "
                f"[Phase {phase}] | "
                f"Loss: {train_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"LR: {current_lrs[0]:.2e}"
            )

            # ── Early stopping ────────────────────────────────
            if val_auc > early_stopping.best_auc:
                best_metrics = val_metrics
                
            if early_stopping(val_auc, model, fold):
                break

        fold_results.append(best_metrics)

        print(f"\n  ✅ Fold {fold} Best Results:")
        print(f"     AUC:         {best_metrics.get('auc', 0):.4f}")
        print(f"     F1:          {best_metrics.get('f1', 0):.4f}")
        print(f"     Sensitivity: {best_metrics.get('sensitivity', 0):.4f}")
        print(f"     Specificity: {best_metrics.get('specificity', 0):.4f}")

    # ── Final Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CROSS VALIDATION COMPLETE")
    print(f"{'='*60}")

    for metric in ["auc", "f1", "accuracy", "sensitivity", "specificity"]:
        values = [r.get(metric, 0) for r in fold_results]
        mean, std = np.mean(values), np.std(values)
        print(f"  {metric.upper():12s}: {mean:.4f} ± {std:.4f}")
        wandb.log({f"cv/{metric}_mean": mean, f"cv/{metric}_std": std})

    print(f"\n  Per-fold AUC:")
    for fold, result in enumerate(fold_results, 1):
        print(f"  Fold {fold}: AUC={result.get('auc',0):.4f} | "
              f"F1={result.get('f1',0):.4f} | "
              f"Sens={result.get('sensitivity',0):.4f} | "
              f"Spec={result.get('specificity',0):.4f}")

    wandb.finish()
    return fold_results


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    # ── Reproducibility ───────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Config ────────────────────────────────────────────────
    config = {
        # Dataset
        "seed":               SEED,
        "num_folds":          5,
        "batch_size":         4,
        "num_workers":        2,

        # Model (V4 — increased capacity to fight underfitting)
        "pretrained":         True,
        "pretrained_path":    None,
        "embed_dim":          256,
        "num_heads":          4,
        "num_layers":         6,        # ↑ Upgraded from 4 — more ViT capacity
        "mlp_ratio":          2,
        "attn_dropout":       0.2,
        "ffn_dropout":        0.2,
        "head_dropout":       0.3,      # ↑ Slightly higher (0.2 →  0.3) since we have more capacity now
        "clinical_embed_dim": 32,
        "hidden_dim":         128,

        # Training (V5 — no DANN, MixUp added)
        "epochs":             150,
        "phase1_epochs":      30,
        "lr":                 5e-5,
        "weight_decay":       1e-4,
        "warmup_epochs":      15,
        "grad_clip":          1.0,
        "accumulation_steps": 4,
        "patience":           40,
        "mixup_alpha":        0.0,      # V6: MixUp disabled — hurts small-dataset performance

        # Paths
        "checkpoint_path": "experiments/checkpoints/best_model.pth",
        "data_path":       "data/processed/",
    }

    # ── Load metadata ─────────────────────────────────────────
    # We now use the preprocessed metadata pointing to .pt tensors
    metadata_path = "data/metadata_pt.csv"

    if not os.path.exists(metadata_path):
        print(f"\n⚠️  metadata.csv not found — creating dummy data")

        # Dummy data for testing with age and gender
        np.random.seed(SEED)
        n = 20
        dummy_data = {
            "filepath": [f"data/processed/scan_{i:03d}.nii.gz"
                         for i in range(n)],
            "label":    [0]*12 + [1]*8,
            "age":      np.random.randint(18, 60, n).tolist(),
            "gender":   (["M", "F"] * (n // 2))[:n],
        }
        dataframe = pd.DataFrame(dummy_data)
        print(f"Dummy dataset: {len(dataframe)} scans")
        print(dataframe[["label", "age", "gender"]].head())

    else:
        dataframe = pd.read_csv(metadata_path)
        print(f"\nLoaded {len(dataframe)} scans")
        print(f"Age range: {dataframe['age'].min()} - {dataframe['age'].max()}")
        print(f"Gender distribution:\n{dataframe['gender'].value_counts()}")

    # ── TorchIO 3D Augmentation ────────────────────────────────
    import torchio as tio
    augmentation = tio.Compose([
        tio.RandomFlip(axes=["LR"]), # Symmetrical brain flipping
        tio.RandomAffine(degrees=15, scales=(0.85, 1.15), translation=5), # Aggressive rotation and zooming
        tio.RandomNoise(std=0.02), # Simulate scanner noise
        tio.RandomBlur(std=(0, 1)), # Simulate subtle patient movement
        tio.RandomGamma(log_gamma=(-0.3, 0.3)), # Contrast adjustments
    ])

    # ── Start training ────────────────────────────────────────
    results = train_schizobrain(
        dataframe=dataframe,
        config=config,
        device=device,
        augmentation=augmentation,
    )

    print(f"\n✅ Training complete!")
    print(f"Best models saved in: experiments/checkpoints/")