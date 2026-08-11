import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import wandb
from tqdm import tqdm
import pandas as pd

from src.model.hybrid import SchizoBrain
from src.data.transforms import AgeNormalizer
from src.data.dataset import MRIDataset
from src.training.loss import FocalLoss
from src.training.utils import WarmupCosineScheduler, EarlyStopping
from src.training.engine import train_epoch, validate_epoch

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