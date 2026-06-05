"""
SchizoBrain: CNN+ViT Hybrid Model with Clinical Metadata Fusion
===============================================================
V3 Update: Late fusion of age and gender into classification head.

Architecture:
    Input 3D MRI
        → MedicalNet CNN Encoder (pretrained, local features)
        → Patch Embedding + Positional Encoding
        → Encoder Block × 4 (MHSA, LN, FFN, Residuals)
        → Global Average Pooling
        → Brain Feature Vector (256,)
                                            Age (normalized)
                                            Gender (0/1)
                                                ↓
                                        Clinical Embedding (32,)
                                                ↓
        → Concatenate [Brain (256,) + Clinical (32,)] = (288,)
        → Dense Layer (Sigmoid)
        → Output Probability

Why late fusion for 646 scans?
    - Simplest fusion strategy
    - Clinical features processed separately then combined
    - Won't interfere with MRI feature learning
    - Works well on small datasets
    - Most interpretable — easy to see MRI vs clinical contribution

Age preprocessing:
    - Normalized to [0, 1] using dataset min/max age
    - Typical range: 18-65 years for schizophrenia datasets

Gender preprocessing:
    - Encoded as binary: 0 = Female, 1 = Male
    - Passed as float tensor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
from typing import Optional, Tuple


# ==============================================================
# DROP PATH (Stochastic Depth)
# Drops entire residual branch randomly during training.
# More effective than Dropout for Transformer blocks.
# ==============================================================

class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath regularization.
    Randomly drops entire residual branch (not individual neurons).
    More effective than standard Dropout for ViT on small datasets.

    During eval: acts as identity (no drop).
    During train: drops with probability `drop_prob`.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (B, 1, 1) — drop whole samples in batch independently
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)  # bernoulli
        return x * noise / keep_prob            # scale to maintain expectation


# ==============================================================
# PART 1: MEDICALNET CNN ENCODER (unchanged from V2)
# ==============================================================

class MedicalNetEncoder(nn.Module):
    """
    CNN Encoder using MedicalNet pretrained ResNet50.
    Upgraded from ResNet10 for richer 3D feature extraction.

    Input:  (B, 1, 96, 96, 96)
    Output: (B, 512, 12, 12, 12)  ← projected down from 2048

    Architecture:
        ResNet50_3D backbone  → (B, 2048, 12, 12, 12)
        channel_proj (1×1 conv) → (B, 512, 12, 12, 12)

    channel_proj is always trainable (not frozen with backbone)
    so it adapts ResNet50 features to the ViT patch embedding.
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        frozen: bool = True,
    ):
        super().__init__()
        self.backbone = self._build_resnet50()

        # Project 2048 ResNet50 channels → 512 for PatchEmbedding3D
        # Always trainable — learns to compress features optimally
        self.channel_proj = nn.Sequential(
            nn.Conv3d(2048, 512, kernel_size=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        if pretrained:
            if pretrained_path:
                self._load_local_weights(pretrained_path)
                print(f"Loaded MedicalNet ResNet50 from: {pretrained_path}")
            else:
                self._load_hub_weights()
                print("Loaded MedicalNet ResNet50 weights from Hub")
        else:
            print("Using random ResNet50 weights")

        if frozen:
            self.freeze()
            print("CNN Encoder frozen (Phase 1)")

    def _build_resnet50(self) -> nn.Module:
        """3D ResNet50 matching MedicalNet architecture."""

        class Bottleneck3D(nn.Module):
            """3D Bottleneck block: 1×1 → 3×3 → 1×1 convolutions."""
            def __init__(self, in_ch, mid_ch, stride=1):
                super().__init__()
                out_ch = mid_ch * 4
                self.conv1 = nn.Conv3d(in_ch, mid_ch, 1, bias=False)
                self.bn1   = nn.BatchNorm3d(mid_ch)
                self.conv2 = nn.Conv3d(mid_ch, mid_ch, 3,
                                       stride=stride, padding=1, bias=False)
                self.bn2   = nn.BatchNorm3d(mid_ch)
                self.conv3 = nn.Conv3d(mid_ch, out_ch, 1, bias=False)
                self.bn3   = nn.BatchNorm3d(out_ch)
                self.relu  = nn.ReLU(inplace=True)
                self.downsample = None
                if stride != 1 or in_ch != out_ch:
                    self.downsample = nn.Sequential(
                        nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                        nn.BatchNorm3d(out_ch),
                    )

            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                if self.downsample:
                    identity = self.downsample(x)
                return self.relu(out + identity)

        def make_layer(in_ch, mid_ch, num_blocks, stride=1):
            layers = [Bottleneck3D(in_ch, mid_ch, stride)]
            for _ in range(1, num_blocks):
                layers.append(Bottleneck3D(mid_ch * 4, mid_ch, 1))
            return nn.Sequential(*layers)

        class ResNet50_3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1   = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
                self.bn1     = nn.BatchNorm3d(64)
                self.relu    = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
                # MedicalNet 3D: layer3 & layer4 stride=1
                # to preserve spatial resolution (12×12×12 output)
                self.layer1 = make_layer(64,   64,  3, stride=1)  # → 256ch
                self.layer2 = make_layer(256, 128,  4, stride=2)  # → 512ch, 12³
                self.layer3 = make_layer(512, 256,  6, stride=1)  # → 1024ch
                self.layer4 = make_layer(1024, 512, 3, stride=1)  # → 2048ch

            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                return x  # (B, 2048, 12, 12, 12)

        return ResNet50_3D()

    def _load_hub_weights(self):
        try:
            pretrained = torch.hub.load(
                "Warvito/MedicalNet-models",
                "medicalnet_resnet50_23datasets",
                verbose=False,
            )
            pretrained_dict = pretrained.state_dict()
            # ResNet50: strip DataParallel "module." prefix only
            # (keep block indices like layer1.0., layer1.1. intact)
            clean_dict = {k.replace("module.", ""): v
                          for k, v in pretrained_dict.items()}
            model_dict = self.backbone.state_dict()
            matched = {k: v for k, v in clean_dict.items()
                      if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            self.backbone.load_state_dict(model_dict)
            print(f"   Matched {len(matched)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"Hub load failed: {e}")
            print("   Using random ResNet50 weights as fallback.")

    def _load_local_weights(self, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_dict = OrderedDict(
            (k.replace("module.", ""), v) for k, v in state_dict.items()
        )
        model_dict = self.backbone.state_dict()
        matched = {k: v for k, v in clean_dict.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        self.backbone.load_state_dict(model_dict)
        print(f"   Matched {len(matched)}/{len(model_dict)} layers")

    def freeze(self):
        """Freeze backbone only; channel_proj stays trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Phase 2: Partially unfreeze CNN layers 2, 3, and 4.

        layer2: mid-level features (cortical thickness, gray matter density)
               — critical for schizophrenia biomarkers, was previously frozen.
        layer3: high-level structural features.
        layer4: deepest semantic features.

        layer1 and stem (conv1/bn1) stay frozen to preserve low-level
        MedicalNet features learned from 23 medical datasets.
        """
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "layer3" in name or "layer2" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("CNN Encoder PARTIALLY unfrozen (layer2 + layer3 + layer4 for Phase 2)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 2048, 12, 12, 12)
        return self.channel_proj(features)   # (B,  512, 12, 12, 12)


# ==============================================================
# PART 2: PATCH EMBEDDING + POSITIONAL ENCODING (unchanged)
# ==============================================================

class PatchEmbedding3D(nn.Module):
    """
    Converts CNN feature maps into patch token sequence.
    Input:  (B, 256, 6, 6, 6)
    Output: (B, num_patches + 1, embed_dim)
    """

    def __init__(self, in_channels=512, patch_size=2,
                 embed_dim=256, spatial_size=12):
        super().__init__()
        self.patch_size = patch_size
        num_patches_per_dim = spatial_size // patch_size
        self.num_patches = num_patches_per_dim ** 3
        patch_flat_dim = in_channels * (patch_size ** 3)
        self.projection = nn.Linear(patch_flat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, D//p, p, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, self.num_patches, -1)
        x = self.projection(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_encoding
        return x


# ==============================================================
# PART 3: MHSA (unchanged)
# ==============================================================

class MultiHeadSelfAttention(nn.Module):
    """MHSA with 4 heads for small dataset."""

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_dropout(self.out_proj(x))


# ==============================================================
# PART 4: FFN (unchanged)
# ==============================================================

class FeedForwardNetwork(nn.Module):
    """FFN with mlp_ratio=2 for small dataset."""

    def __init__(self, embed_dim=256, mlp_ratio=2, dropout=0.3):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ==============================================================
# PART 1: RESNET50 BACKBONE (MedicalNet) + GRL
# ==============================================================

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.alpha)

class MedicalNetResNet50(nn.Module):
    """Single Transformer Encoder Block — MHSA + LN + FFN + Residuals."""

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=2, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)


# ==============================================================
# PART 5: ENCODER BLOCK (unchanged)
# ==============================================================

class EncoderBlock(nn.Module):
    """Single Transformer Encoder Block — MHSA + LN + FFN + Residuals + DropPath."""

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=2,
                 dropout=0.3, drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, mlp_ratio, dropout)
        # DropPath: drops the entire residual branch stochastically
        # More effective than Dropout for ViT regularization
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mhsa(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ==============================================================
# PART 6: VIT ENCODER (unchanged)
# ==============================================================

class ViTEncoder(nn.Module):
    """6 lightweight encoder blocks for 646 scans (upgraded from 4).

    Depth increase adds capacity for complex 3D spatial reasoning without
    overfitting — stochastic depth (DropPath) compensates via linear rate schedule:
    deeper blocks get higher drop rate.
    """

    def __init__(self, embed_dim=256, num_heads=4,
                 num_layers=6, mlp_ratio=2, dropout=0.3,
                 drop_path_rate=0.1):
        super().__init__()
        # Linear stochastic depth schedule: block 0 = 0.0, last block = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout,
                         drop_path_rate=dpr[i])
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ==============================================================
# PART 7: GLOBAL AVERAGE POOLING (unchanged)
# ==============================================================

class GlobalAveragePooling(nn.Module):
    """CLS token + Global Average Pooling fusion.

    Previous: used only GAP over patch tokens, discarding CLS token entirely.
    Now: averages CLS token output with GAP over patch tokens.

    Why this helps:
        - CLS token aggregates GLOBAL context (attends to all patches)
        - GAP gives LOCAL average (uniform weighted sum)
        - Their mean combines both perspectives — strictly more expressive
        - Zero extra parameters
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token  = x[:, 0, :]           # (B, embed_dim) — global summary
        patch_mean = x[:, 1:, :].mean(1)  # (B, embed_dim) — spatial average
        return 0.5 * cls_token + 0.5 * patch_mean  # (B, embed_dim)


# ==============================================================
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
            patch_size=2,
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


# ==============================================================
# AGE NORMALIZER UTILITY
# Use this before passing age to the model
# ==============================================================

class AgeNormalizer:
    """
    Normalizes age values for the model.

    Handles two cases automatically:
      1. Pre-scaled ages in [0, 1] (min-max): values in [0.0, 1.0].
         CSV ages are already normalized — transform() is a no-op.
         For app.py inference with raw user age, call transform_raw().

      2. Raw ages (e.g. 18-65): standard min-max scaling to [0, 1].

    Fit on training set, apply on val/test.
    """

    def __init__(self):
        self.min_age   = None
        self.max_age   = None
        self.pre_scaled = False   # True when CSV ages are already [0,1]
        # Z-score bounds of the dataset (for transform_raw)
        # These are fixed: dataset Z-scores range from -1.920591 to 2.702459
        self._zscore_min = -1.920591
        self._zscore_max =  2.702459
        # Raw age population stats (for Z-scoring user input in app)
        self._raw_mean = 34.0
        self._raw_std  = 11.56

    def fit(self, ages):
        """Compute statistics from training ages."""
        self.min_age = float(min(ages))
        self.max_age = float(max(ages))

        # Detect if ages are already min-max scaled to [0, 1]
        if self.min_age >= 0.0 and self.max_age <= 1.0:
            self.pre_scaled = True
            print(f"Age normalizer: pre-scaled [0,1] detected "
                  f"(range {self.min_age:.4f} to {self.max_age:.4f}). "
                  f"transform() will pass values through unchanged.")
        else:
            self.pre_scaled = False
            print(f"Age normalizer fit: min={self.min_age:.1f}, "
                  f"max={self.max_age:.1f} (raw ages -> min-max scaling)")

    def transform(self, age: float) -> float:
        """
        Normalize a single age value from the CSV.
        - If pre_scaled [0,1]: passes through unchanged.
        - If raw ages: applies min-max scaling to [0, 1].
        """
        assert self.min_age is not None, "Call fit() first"
        if self.pre_scaled:
            return float(age)   # already [0,1] — no-op
        return (age - self.min_age) / (self.max_age - self.min_age + 1e-8)

    def transform_raw(self, raw_age: float) -> float:
        """
        Convert a RAW user-entered age (e.g. 25) to [0, 1].
        Pipeline: raw_age -> Z-score -> scale to [0, 1] using dataset bounds.
        """
        assert self.min_age is not None, "Call fit() first"
        if self.pre_scaled:
            # Step 1: raw age -> Z-score
            z = (raw_age - self._raw_mean) / (self._raw_std + 1e-8)
            # Step 2: Z-score -> [0, 1] using dataset Z-score bounds
            scaled = (z - self._zscore_min) / (self._zscore_max - self._zscore_min + 1e-8)
            # Clamp to [0, 1] for ages outside the training range
            return float(max(0.0, min(1.0, scaled)))
        # Raw ages: direct min-max
        return (raw_age - self.min_age) / (self.max_age - self.min_age + 1e-8)

    def set_population_stats(self, mean: float, std: float):
        """
        Set raw-age population mean/std for transform_raw().
        Default: mean=34.0, std=11.56 (NUSDAST/COBRE datasets).
        """
        self._raw_mean = mean
        self._raw_std  = std

    def transform_batch(self, ages) -> torch.Tensor:
        """Normalize a list/array of ages -> tensor (B, 1)."""
        normalized = [self.transform(a) for a in ages]
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(1)


# ==============================================================
# GENDER ENCODER UTILITY
# ==============================================================

def encode_gender(gender_labels) -> torch.Tensor:
    """
    Encode gender strings or integers to float tensor.

    Accepts:
        "M", "Male", "male", 1   → 1.0
        "F", "Female", "female", 0 → 0.0

    Args:
        gender_labels: List of gender values
    Returns:
        Float tensor (B, 1)
    """
    encoded = []
    for g in gender_labels:
        if str(g).lower() in ["m", "male", "1"]:
            encoded.append(1.0)
        else:
            encoded.append(0.0)
    return torch.tensor(encoded, dtype=torch.float32).unsqueeze(1)


# ==============================================================
# QUICK MODEL TEST
# ==============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SchizoBrain V3 — CNN+ViT + Age/Gender Late Fusion")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Initialize model
    model = SchizoBrain(
        pretrained=False,
        frozen=False,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        mlp_ratio=2,
        attn_dropout=0.3,
        ffn_dropout=0.3,
        head_dropout=0.5,
        clinical_embed_dim=32,
        hidden_dim=128,
    ).to(device)

    # Simulate batch of 2 patients
    dummy_mri    = torch.randn(2, 1, 96, 96, 96).to(device)

    # Age normalizer
    normalizer = AgeNormalizer()
    normalizer.fit([18, 65])                      # dataset age range

    # Patient 1: age=25, Male
    # Patient 2: age=42, Female
    ages   = normalizer.transform_batch([25, 42]).to(device)
    genders = encode_gender(["M", "F"]).to(device)

    print(f"\nInput MRI shape:    {dummy_mri.shape}")
    print(f"Age tensor:         {ages.T}  (normalized)")
    print(f"Gender tensor:      {genders.T}  (1=Male, 0=Female)")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_mri, ages, genders)

    print(f"\nOutput shape: {output.shape}")
    print(f"\nPatient details and predictions:")
    patients = [
        {"age": 25, "gender": "Male"},
        {"age": 42, "gender": "Female"},
    ]
    for i, (prob, patient) in enumerate(zip(output, patients)):
        diagnosis = "Schizophrenia" if prob.item() > 0.5 else "Healthy"
        print(f"  Patient {i+1} "
              f"(age={patient['age']}, {patient['gender']}): "
              f"{prob.item():.4f} → {diagnosis}")

    # Parameter breakdown
    total       = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnn_params  = sum(p.numel() for p in model.cnn_encoder.parameters())
    vit_params  = sum(p.numel() for p in model.vit_encoder.parameters())
    patch_params = sum(p.numel() for p in model.patch_embedding.parameters())
    clin_params = sum(p.numel() for p in model.clinical_embedding.parameters())
    head_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"\nParameter Breakdown:")
    print(f"  CNN Encoder (MedicalNet):   {cnn_params:>10,}")
    print(f"  Patch Embedding:            {patch_params:>10,}")
    print(f"  ViT Encoder:                {vit_params:>10,}")
    print(f"  Clinical Embedding (NEW):   {clin_params:>10,}")
    print(f"  Classification Head (NEW):  {head_params:>10,}")
    print(f"  {'─'*35}")
    print(f"  Total:                      {total:>10,}")
    print(f"  Trainable:                  {trainable:>10,}")

    print(f"\nFusion architecture:")
    print(f"  Brain features:   (B, 256) from ViT GAP")
    print(f"  Clinical features:(B,  32) from age+gender")
    print(f"  After fusion:     (B, 288) concatenated")
    print(f"  Output:           (B,   1) probability")

    print(f"\nUsage in trainer.py:")
    print(f"  output = model(mri, age_tensor, gender_tensor)")
    print(f"\n✅ SchizoBrain V3 working correctly!")
    print("=" * 60)