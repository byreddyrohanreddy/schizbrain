import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

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


