import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

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


