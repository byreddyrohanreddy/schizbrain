import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from typing import Optional, Tuple

from .blocks import *

# PART 8: CLINICAL METADATA EMBEDDING  ← NEW
# Encodes age and gender into a small embedding vector
# ==============================================================

class ClinicalEmbedding(nn.Module):
    """
    Encodes age and gender into a clinical feature vector.

    Input:
        age:    Normalized age float (B, 1)  — value between 0 and 1
        gender: Binary gender float (B, 1)   — 0=Female, 1=Male

    Processing:
        [age, gender] → (B, 2)
            ↓
        Linear(2 → 16) → ReLU → Dropout
            ↓
        Linear(16 → 32) → ReLU
            ↓
        Clinical embedding (B, 32)

    Why separate embedding?
        - Age and gender are very different from MRI voxel values
        - A dedicated small network learns their clinical meaning
        - Then fused with brain features at classification head

    Age normalization (do this before passing to model):
        age_normalized = (age - dataset_min_age) /
                         (dataset_max_age - dataset_min_age)

    Gender encoding:
        Female = 0.0
        Male   = 1.0
    """

    def __init__(
        self,
        clinical_embed_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 2 inputs: age (1) + gender (1)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, clinical_embed_dim),
            nn.ReLU(),
        )

        # Initialize weights small — clinical features
        # should not dominate brain features early in training
        for layer in self.clinical_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            age:    Normalized age (B, 1) — must be between 0 and 1
            gender: Binary gender (B, 1) — 0.0=Female, 1.0=Male
        Returns:
            Clinical embedding (B, 32)
        """
        # Concatenate age and gender: (B, 2)
        clinical_input = torch.cat([age, gender], dim=1)

        # Embed into clinical feature vector: (B, 32)
        return self.clinical_mlp(clinical_input)


# ==============================================================
# PART 9: CLASSIFICATION HEAD WITH LATE FUSION  ← UPDATED
# Brain features + Clinical features fused here
# ==============================================================

class ClassificationHeadWithFusion(nn.Module):
    """
    Late Fusion Classification Head.

    Receives:
        brain_features:    From ViT GAP output (B, embed_dim=256)
        clinical_features: From ClinicalEmbedding (B, 32)

    Fusion:
        Concatenate → (B, 256 + 32) = (B, 288)
            ↓
        Linear(288 → 128) → GELU → Dropout(0.5)
            ↓
        Linear(128 → 1)
            ↓
        Output Logits (B, 1)

    Why concatenate at this stage?
        - Brain features are fully processed by ViT
        - Clinical features add complementary information
        - Model learns how to weight MRI vs age/gender
          for the final schizophrenia decision

    Clinical contribution example:
        MRI alone:          "85% probability schizophrenia"
        MRI + age=22 male:  "91% probability schizophrenia"
        MRI + age=55 female:"79% probability schizophrenia"
        (age and gender modulate the final decision)
    """

    def __init__(
        self,
        brain_dim: int = 256,
        clinical_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Total dimension after concatenation
        fused_dim = brain_dim + clinical_dim    # 256 + 32 = 288

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),   # 288 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),           # 128 → 1
        )

    def forward(
        self,
        brain_features: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            brain_features:    (B, 256) from GAP
            clinical_features: (B, 32)  from ClinicalEmbedding
        Returns:
            Schizophrenia logits (B, 1)
        """
        # Late fusion: concatenate brain + clinical
        fused = torch.cat([brain_features, clinical_features], dim=1)  # (B, 288)

        # Final classification
        return self.head(fused)                 # (B, 1)


# ==============================================================
# PART 10: SCHIZOBRAIN V3 — FULL MODEL WITH METADATA FUSION
# ==============================================================

class SchizoBrain(nn.Module):
    """
    SchizoBrain V3: CNN+ViT + Age/Gender Late Fusion.

    Changes from V2:
        + ClinicalEmbedding module for age and gender
        + ClassificationHeadWithFusion replaces ClassificationHead
        + forward() now takes age and gender as additional inputs

    Forward pass inputs:
        mri:    3D MRI volume  (B, 1, 96, 96, 96)
        age:    Normalized age (B, 1)  — float between 0 and 1
        gender: Binary gender  (B, 1)  — 0.0=Female, 1.0=Male

    How to normalize age before passing:
        age_norm = (age - min_age) / (max_age - min_age)
        # e.g. age=25, min=18, max=65
        # age_norm = (25-18)/(65-18) = 0.149

    How to encode gender:
        gender = 0.0  # Female
        gender = 1.0  # Male
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        frozen: bool = True,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: int = 2,
        attn_dropout: float = 0.3,
        ffn_dropout: float = 0.3,
        head_dropout: float = 0.5,
        clinical_embed_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # ── CNN Encoder (MedicalNet pretrained) ──────────────
        self.cnn_encoder = MedicalNetEncoder(
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            frozen=frozen,
        )

        # 🧩 Patch Embedding + Positional Encoding 🧩
        self.patch_embedding = PatchEmbedding3D(
            in_channels=512,
            patch_size=4,  # Changed from 2 to match checkpoint training
            embed_dim=embed_dim,
            spatial_size=12,
        )

        # ── ViT Encoder (6 blocks) ────────────────────────────
        self.vit_encoder = ViTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=attn_dropout,
            drop_path_rate=0.1,  # linear schedule 0→0.1 across blocks
        )

        # ── Global Average Pooling ────────────────────────────
        self.gap = GlobalAveragePooling()

        # ── Clinical Embedding (age + gender) ← NEW ──────────
        self.clinical_embedding = ClinicalEmbedding(
            clinical_embed_dim=clinical_embed_dim,
            dropout=head_dropout,
        )

        # ── Classification Head with Late Fusion ← UPDATED ───
        self.classifier = ClassificationHeadWithFusion(
            brain_dim=embed_dim,                # 256
            clinical_dim=clinical_embed_dim,    # 32
            hidden_dim=hidden_dim,              # 128
            dropout=head_dropout,
        )

        # ── Site Harmonization (DANN) ─────────────────────────
        # 4 sites: COBRE(0), NUSDAST(1), ABIDE(2), CC(3)
        self.grl = GradientReversal(alpha=1.0)
        self.site_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 4)
        )

    def forward(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        return_site: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass with clinical metadata.

        Args:
            mri:         Raw 3D MRI volume (B, 1, 96, 96, 96)
            age:         Normalized age    (B, 1)
            gender:      Binary gender     (B, 1)
            return_site: If True, also returns site_logits

        Returns:
            Schizophrenia logits (B, 1)
            (Optional) site_logits (B, 4) if return_site=True
        """

        # ── MRI pathway ───────────────────────────────────────
        brain = self.cnn_encoder(mri)          # (B, 256, 6, 6, 6)
        brain = self.patch_embedding(brain)    # (B, 28, 256)
        brain = self.vit_encoder(brain)        # (B, 28, 256)
        brain = self.gap(brain)                # (B, 256)

        # ── Clinical pathway ──────────────────────────────────
        clinical = self.clinical_embedding(    # (B, 32)
            age, gender
        )

        # ── Late fusion + classification ──────────────────────
        output = self.classifier(             # (B, 1)
            brain, clinical
        )

        if return_site:
            site_logits = self.site_classifier(self.grl(brain))
            return output, site_logits

        return output

    def phase1_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-4):
        """Phase 1: channel_proj + ViT + clinical embedding + classifier train.
        CNN backbone stays frozen."""
        params = (
            list(self.cnn_encoder.channel_proj.parameters()) +  # always trainable
            list(self.patch_embedding.parameters()) +
            list(self.vit_encoder.parameters()) +
            list(self.gap.parameters()) +
            list(self.clinical_embedding.parameters()) +
            list(self.classifier.parameters()) +
            list(self.site_classifier.parameters())
        )
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def phase2_optimizer(self, vit_lr: float = 1e-4, cnn_lr: float = 1e-5, weight_decay: float = 1e-4):
        """Phase 2: CNN backbone fine-tunes at smaller LR, everything else normal."""
        return torch.optim.AdamW([
            {
                "params": self.cnn_encoder.backbone.parameters(),
                "lr": cnn_lr,                  # 1e-5 — careful fine-tuning
            },
            {
                "params": (
                    list(self.cnn_encoder.channel_proj.parameters()) +
                    list(self.patch_embedding.parameters()) +
                    list(self.vit_encoder.parameters()) +
                    list(self.gap.parameters()) +
                    list(self.clinical_embedding.parameters()) +
                    list(self.classifier.parameters()) +
                    list(self.site_classifier.parameters())
                ),
                "lr": vit_lr,                  # 1e-4 — normal
            },
        ], weight_decay=weight_decay)

    def switch_to_phase2(self):
        """Unfreeze CNN for Phase 2 fine-tuning."""
        self.cnn_encoder.unfreeze()


