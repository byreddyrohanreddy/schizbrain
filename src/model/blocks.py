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


