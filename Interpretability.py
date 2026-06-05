"""
SchizoBrain Interpretability
==============================
Grad-CAM + Attention Maps for CNN+ViT Hybrid Model.

Two complementary interpretability methods:

1. GRAD-CAM (from CNN layers)
   - Shows WHICH brain regions the CNN focused on
   - Produces a 3D heatmap overlaid on MRI slices
   - Works on CNN layer4 (last ResNet block)
   - Best for: spatial localization of brain abnormalities

2. ATTENTION MAPS (from ViT blocks)
   - Shows WHICH patches the ViT attended to globally
   - Captures long-range brain region relationships
   - Uses attention rollout across all 4 encoder blocks
   - Best for: understanding global brain connectivity patterns

Together they give the doctor:
    "The CNN found abnormalities in the prefrontal cortex
     AND the ViT connected those to distant temporal regions"

Usage:
    interpreter = SchizoBrainInterpreter(model)
    results = interpreter.explain(mri, age, gender)
    interpreter.visualize(mri, results, save_path="report.png")
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ==============================================================
# GRAD-CAM ENGINE
# Extracts gradients from CNN layer4 to build heatmap
# ==============================================================

class GradCAM3D:
    """
    Grad-CAM for 3D CNN layers in SchizoBrain.

    How it works:
        1. Register hooks on target CNN layer (layer4)
        2. Run forward pass → get prediction
        3. Backpropagate prediction score w.r.t. feature maps
        4. Weight each feature map by its gradient (importance)
        5. Sum weighted maps → ReLU → upsample to MRI size
        6. Result: 3D heatmap showing important brain regions

    Target layer: model.cnn_encoder.backbone.layer4
    (last ResNet block — highest level spatial features)

    Args:
        model:        Trained SchizoBrain V3 model
        target_layer: CNN layer to extract Grad-CAM from
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for hooks
        self._feature_maps = None   # forward hook result
        self._gradients = None      # backward hook result

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(
            self._save_feature_maps
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_feature_maps(self, module, input, output):
        """Forward hook — saves CNN feature maps."""
        self._feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook — saves gradients flowing into layer."""
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for one MRI scan.

        Args:
            mri:          MRI volume (1, 1, 96, 96, 96) — single scan
            age:          Normalized age (1, 1)
            gender:       Binary gender (1, 1)
            target_class: 1=Schizophrenia, 0=Healthy
        Returns:
            gradcam_map: 3D heatmap (96, 96, 96) — same size as MRI
                         Values 0-1, higher = more important region
        """
        self.model.eval()

        # Enable gradients for Grad-CAM
        mri = mri.clone().requires_grad_(True)

        # Forward pass
        output = self.model(mri, age, gender)    # (1, 1) — raw logits
        probability = torch.sigmoid(output[0, 0]).item()  # convert logit → probability

        # Score to backpropagate
        # For schizophrenia: maximize output
        # For healthy: minimize output (1 - output)
        if target_class == 1:
            score = output[0, 0]
        else:
            score = 1 - output[0, 0]

        # Zero gradients and backpropagate
        self.model.zero_grad()
        score.backward()

        # ── Compute Grad-CAM ──────────────────────────────────

        # Gradients: (1, C, D, H, W)
        gradients = self._gradients             # (1, 256, 6, 6, 6)

        # Feature maps: (1, C, D, H, W)
        feature_maps = self._feature_maps       # (1, 256, 6, 6, 6)

        # Global average pool gradients → importance weights per channel
        # Shape: (1, C, 1, 1, 1)
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weight each feature map by its importance
        # Shape: (1, C, D, H, W)
        weighted_maps = weights * feature_maps

        # Sum across channels → (1, 1, D, H, W)
        cam = weighted_maps.sum(dim=1, keepdim=True)

        # ReLU — keep only regions that push toward target class
        cam = F.relu(cam)

        # Upsample to original MRI size (96, 96, 96)
        cam = F.interpolate(
            cam,
            size=(96, 96, 96),
            mode="trilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, probability

    def remove_hooks(self):
        """Remove hooks when done — important for memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ==============================================================
# GRAD-CAM++ ENGINE
# Better spatial localization than standard Grad-CAM
# ==============================================================

class GradCAMPlusPlus3D:
    """
    Grad-CAM++ for 3D CNN — better localization than Grad-CAM.

    Improvement over Grad-CAM:
        - Uses second-order gradients
        - More precise spatial localization
        - Better when multiple regions are important
        - Recommended for clinical use

    For schizophrenia: multiple brain regions affected simultaneously
    → Grad-CAM++ highlights all of them more accurately

    Args:
        model:        Trained SchizoBrain V3 model
        target_layer: CNN layer to extract Grad-CAM++ from
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._feature_maps = None
        self._gradients = None

        self._forward_hook = target_layer.register_forward_hook(
            self._save_feature_maps
        )
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_feature_maps(self, module, input, output):
        self._feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def compute(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Grad-CAM++ heatmap.

        Args:
            mri:          MRI volume (1, 1, 96, 96, 96)
            age:          Normalized age (1, 1)
            gender:       Binary gender (1, 1)
            target_class: 1=Schizophrenia, 0=Healthy
        Returns:
            gradcam_pp:  3D heatmap (96, 96, 96)
            probability: Model prediction probability
        """
        self.model.eval()
        mri = mri.clone().requires_grad_(True)

        output = self.model(mri, age, gender)
        probability = torch.sigmoid(output[0, 0]).item()  # convert logit → probability

        score = output[0, 0] if target_class == 1 else 1 - output[0, 0]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # ── Grad-CAM++ weights ────────────────────────────────
        gradients  = self._gradients      # (1, C, D, H, W)
        feat_maps  = self._feature_maps   # (1, C, D, H, W)

        # Second order gradient approximation
        grad_sq    = gradients ** 2
        grad_cube  = gradients ** 3

        # Denominator for alpha computation
        sum_feat   = feat_maps.sum(dim=(2, 3, 4), keepdim=True)
        denom      = 2 * grad_sq + sum_feat * grad_cube + 1e-8

        # Alpha weights (Grad-CAM++ specific)
        alpha      = grad_sq / denom

        # Positive gradients only
        pos_grads  = F.relu(score.exp() * gradients)

        # Weights: sum of alpha * positive gradients over spatial dims
        weights    = (alpha * pos_grads).sum(dim=(2, 3, 4), keepdim=True)

        # Weighted combination of feature maps
        cam        = (weights * feat_maps).sum(dim=1, keepdim=True)
        cam        = F.relu(cam)

        # Upsample to MRI size
        cam = F.interpolate(
            cam.detach(),
            size=(96, 96, 96),
            mode="trilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, probability

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()


# ==============================================================
# ATTENTION MAP EXTRACTOR
# Extracts and rolls out ViT attention weights
# ==============================================================

class AttentionMapExtractor:
    """
    Extracts attention maps from all ViT encoder blocks.

    Uses Attention Rollout to propagate attention through
    all 4 encoder blocks — gives the final attention map
    showing which brain patches the model attended to most.

    Attention Rollout algorithm:
        1. Extract attention matrix from each block
        2. Add identity (residual connections)
        3. Multiply matrices across all blocks
        4. Result: final attention from CLS to all patches

    Args:
        model: Trained SchizoBrain V3 model
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._attention_weights = []
        self._hooks = []

        # Register hooks on all MHSA modules in ViT encoder
        for block in model.vit_encoder.blocks:
            hook = block.mhsa.register_forward_hook(
                self._save_attention
            )
            self._hooks.append(hook)

    def _save_attention(self, module, input, output):
        """
        Forward hook on MHSA — saves attention weights.
        We need to recompute attention here since forward()
        doesn't return it separately.
        """
        x = input[0]
        B, N, C = x.shape

        # Recompute attention weights (no dropout for visualization)
        with torch.no_grad():
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, module.head_dim
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = F.softmax(
                (q @ k.transpose(-2, -1)) * module.scale, dim=-1
            )
            # Average across heads: (B, N, N)
            attn = attn.mean(dim=1)
            self._attention_weights.append(attn.detach().cpu())

    def compute_rollout(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        discard_ratio: float = 0.9,
    ) -> np.ndarray:
        """
        Compute attention rollout map.

        Args:
            mri:           MRI volume (1, 1, 96, 96, 96)
            age:           Normalized age (1, 1)
            gender:        Binary gender (1, 1)
            discard_ratio: Drop lowest attention values (noise reduction)
        Returns:
            rollout_map: 3D heatmap (96, 96, 96)
                         Shows which brain regions ViT attended to
        """
        self._attention_weights = []
        self.model.eval()

        with torch.no_grad():
            _ = self.model(mri, age, gender)

        # ── Attention Rollout ─────────────────────────────────
        if not self._attention_weights:
            return np.zeros((96, 96, 96))
            
        N = self._attention_weights[0].shape[1]
        num_patches = N - 1
        grid_size = int(round(num_patches ** (1/3)))

        result = torch.eye(N)  # start with identity

        for attn in self._attention_weights:
            attn_mat = attn[0]  # (N, N) for first batch item

            # Add identity for residual connections
            attn_mat = attn_mat + torch.eye(N)

            # Normalize rows
            attn_mat = attn_mat / attn_mat.sum(dim=-1, keepdim=True)

            # Discard low attention (noise reduction)
            flat = attn_mat.view(-1)
            threshold = flat.kthvalue(
                int(discard_ratio * flat.numel())
            ).values
            attn_mat[attn_mat < threshold] = 0
            attn_mat = attn_mat / (attn_mat.sum(dim=-1, keepdim=True) + 1e-8)

            # Accumulate rollout
            result = attn_mat @ result

        # Extract CLS → patch attention (row 0, columns 1 onwards)
        # Shape: (num_patches,) — one value per patch
        cls_attention = result[0, 1:].numpy()

        # Reshape to 3D patch grid
        patch_grid = cls_attention.reshape(grid_size, grid_size, grid_size)

        # Upsample patch grid back to MRI size (96, 96, 96)
        patch_tensor = torch.tensor(patch_grid).unsqueeze(0).unsqueeze(0)
        attention_map = F.interpolate(
            patch_tensor,
            size=(96, 96, 96),
            mode="trilinear",
            align_corners=False,
        ).squeeze().numpy()

        # Normalize to [0, 1]
        a_min, a_max = attention_map.min(), attention_map.max()
        if a_max - a_min > 1e-8:
            attention_map = (attention_map - a_min) / (a_max - a_min)

        return attention_map

    def get_per_head_attention(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
    ) -> List[np.ndarray]:
        """
        Get attention map per head from last encoder block.
        Useful for visualizing what each head specializes in.

        Returns:
            List of 4 attention maps (one per head)
            Each map shape: (96, 96, 96)
        """
        self.model.eval()
        head_maps = []

        # Hook on last block's MHSA to get per-head attention
        last_block = self.model.vit_encoder.blocks[-1]
        per_head_attn = []

        def hook_fn(module, input, output):
            x = input[0]
            B, N, C = x.shape
            with torch.no_grad():
                qkv = module.qkv(x).reshape(
                    B, N, 3, module.num_heads, module.head_dim
                )
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                attn = F.softmax(
                    (q @ k.transpose(-2, -1)) * module.scale, dim=-1
                )   # (B, num_heads, N, N)
                per_head_attn.append(attn.detach().cpu())

        h = last_block.mhsa.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(mri, age, gender)
        h.remove()

        if per_head_attn:
            attn_all_heads = per_head_attn[0][0]  # (num_heads, N, N)
            N = attn_all_heads.shape[1]
            num_patches = N - 1
            grid_size = int(round(num_patches ** (1/3)))
            
            for head_idx in range(attn_all_heads.shape[0]):
                # CLS token → patch attention for this head
                head_attn = attn_all_heads[head_idx, 0, 1:].numpy()
                patch_grid = head_attn.reshape(grid_size, grid_size, grid_size)
                patch_tensor = torch.tensor(patch_grid).unsqueeze(0).unsqueeze(0)
                head_map = F.interpolate(
                    patch_tensor,
                    size=(96, 96, 96),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze().numpy()

                # Normalize
                h_min, h_max = head_map.min(), head_map.max()
                if h_max - h_min > 1e-8:
                    head_map = (head_map - h_min) / (h_max - h_min)
                head_maps.append(head_map)

        return head_maps

    def remove_hooks(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()


# ==============================================================
# COMBINED INTERPRETER
# Runs both Grad-CAM++ and Attention Rollout together
# ==============================================================

class SchizoBrainInterpreter:
    """
    Combined interpretability engine for SchizoBrain V3.

    Runs both:
        1. Grad-CAM++ on CNN layer4 → spatial heatmap
        2. Attention Rollout on ViT → global attention map
        3. Combined map → weighted average of both

    Usage:
        interpreter = SchizoBrainInterpreter(model)

        results = interpreter.explain(mri, age, gender)
        # results contains: gradcam, attention, combined, probability

        interpreter.visualize(mri, results, save_path="report.png")

        interpreter.cleanup()  # always call when done

    Args:
        model:         Trained SchizoBrain V3 model
        gradcam_weight: Weight for Grad-CAM in combined map (0-1)
        attn_weight:   Weight for attention in combined map (0-1)
    """

    def __init__(
        self,
        model: nn.Module,
        gradcam_weight: float = 0.6,
        attn_weight: float = 0.4,
        temperature: float = 2.0,
    ):
        self.model = model
        self.gradcam_weight = gradcam_weight
        self.attn_weight = attn_weight
        self.temperature = temperature  # > 1.0 softens confidence (calibration)

        # Target layer for Grad-CAM — last CNN block
        target_layer = model.cnn_encoder.backbone.layer4

        # Initialize both engines
        self.gradcam = GradCAM3D(model, target_layer)
        self.attention = AttentionMapExtractor(model)

        print("✅ SchizoBrainInterpreter ready")
        print(f"   Grad-CAM weight: {gradcam_weight}")
        print(f"   Attention weight:  {attn_weight}")

    def _create_brain_mask(self, mri: torch.Tensor) -> np.ndarray:
        """
        Creates a soft brain mask from the MRI intensity.
        Zeros out skull/background regions in the heatmap.

        Strategy:
            - Find the background (which is the minimum value in a normalized tensor)
            - Apply gaussian blur to soften hard edges
            - Erode slightly to pull mask away from skull boundary
        """
        mri_np = mri.squeeze().cpu().numpy()  # (96, 96, 96)

        # In a normalized MRI, the background zeros become the minimum negative value.
        # Anything slightly above this minimum is actual brain tissue.
        background_val = mri_np.min()
        mask = (mri_np > background_val + 1e-3).astype(np.float32)

        # Smooth the mask edges with a simple box blur to avoid hard cutoffs
        from scipy.ndimage import gaussian_filter, binary_erosion
        # Erode slightly to pull mask inward from skull surface
        mask_eroded = binary_erosion(mask, iterations=2).astype(np.float32)
        # Smooth edges so the overlay looks natural
        mask_smooth = gaussian_filter(mask_eroded, sigma=2.0)
        # Re-normalize to [0, 1]
        if mask_smooth.max() > 1e-8:
            mask_smooth = mask_smooth / mask_smooth.max()

        return mask_smooth

    def explain(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        target_class: int = 1,
    ) -> Dict:
        """
        Generate full interpretability report for one scan.

        Args:
            mri:          MRI volume (1, 1, 96, 96, 96)
            age:          Normalized age (1, 1)
            gender:       Binary gender (1, 1)
            target_class: 1=Schizophrenia, 0=Healthy
        Returns:
            Dictionary with:
                gradcam:     Grad-CAM++ heatmap (96, 96, 96)
                attention:   Attention rollout map (96, 96, 96)
                combined:    Weighted combination of both
                probability: Model prediction probability
                diagnosis:   "Schizophrenia" or "Healthy"
                confidence:  Confidence percentage
        """
        device = next(self.model.parameters()).device

        # Move inputs to device
        mri    = mri.to(device)
        age    = age.to(device)
        gender = gender.to(device)

        # ── Brain mask (suppress skull/background) ────────────
        brain_mask = self._create_brain_mask(mri)

        # ── Grad-CAM ──────────────────────────────────────────
        gradcam_map, probability = self.gradcam.compute(
            mri, age, gender, target_class
        )
        # Apply brain mask — zero out non-brain activations
        gradcam_map = gradcam_map * brain_mask
        # Re-normalize after masking
        g_max = gradcam_map.max()
        if g_max > 1e-8:
            gradcam_map = gradcam_map / g_max

        # ── Attention Rollout ─────────────────────────────────
        attention_map = self.attention.compute_rollout(
            mri, age, gender
        )
        # Apply brain mask
        attention_map = attention_map * brain_mask
        a_max = attention_map.max()
        if a_max > 1e-8:
            attention_map = attention_map / a_max

        # ── Combined map ──────────────────────────────────────
        combined_map = (
            self.gradcam_weight * gradcam_map +
            self.attn_weight    * attention_map
        )
        # Renormalize
        c_min, c_max = combined_map.min(), combined_map.max()
        if c_max - c_min > 1e-8:
            combined_map = (combined_map - c_min) / (c_max - c_min)

        # ── Temperature Scaling ───────────────────────────────
        # Convert prob → logit → scale by T → back to prob
        # T > 1.0 softens confidence (calibration for overconfident models)
        # T < 1.0 sharpens confidence (don't use with overconfident models)
        import math
        raw_prob = probability
        if 1e-7 < raw_prob < 1 - 1e-7:
            logit = math.log(raw_prob / (1 - raw_prob))
            scaled_logit = logit / self.temperature
            probability = 1.0 / (1.0 + math.exp(-scaled_logit))

        # ── Diagnosis ─────────────────────────────────────────
        diagnosis  = "Schizophrenia" if probability > 0.5 else "Healthy"
        confidence = probability * 100 if probability > 0.5 \
                     else (1 - probability) * 100
        # Clamp to [1, 99] — 100% certainty is never clinically appropriate
        confidence = max(1.0, min(99.0, confidence))

        return {
            "gradcam":     gradcam_map,
            "attention":   attention_map,
            "combined":    combined_map,
            "probability": probability,
            "diagnosis":   diagnosis,
            "confidence":  confidence,
        }

    def visualize(
        self,
        mri: torch.Tensor,
        results: Dict,
        age_value: float = None,
        gender_value: str = None,
        save_path: Optional[str] = None,
        slices: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Generate clinical visualization report.

        Shows 3 views (axial, coronal, sagittal) for:
            - Original MRI
            - Grad-CAM overlay
            - Attention map overlay
            - Combined map overlay

        Args:
            mri:          MRI volume tensor (1, 1, 96, 96, 96)
            results:      Output from explain()
            age_value:    Patient age (for display)
            gender_value: Patient gender (for display)
            save_path:    Where to save the figure
            slices:       (axial, coronal, sagittal) slice indices
                          Defaults to middle slices
        """
        # Extract MRI numpy array
        mri_np = mri.squeeze().cpu().numpy()  # (96, 96, 96)

        # Default to middle slices
        if slices is None:
            mid = mri_np.shape[0] // 2
            slices = (mid, mid, mid)

        ax_s, cor_s, sag_s = slices

        # Brain heatmap colormap (transparent blue → red)
        colors = [
            (0, 0, 1, 0),      # blue transparent
            (0, 0, 1, 0.3),    # blue semi-transparent
            (0, 1, 0, 0.5),    # green
            (1, 1, 0, 0.7),    # yellow
            (1, 0, 0, 0.9),    # red
        ]
        brain_cmap = LinearSegmentedColormap.from_list(
            "brain_heat", colors, N=256
        )

        # ── Build figure ──────────────────────────────────────
        fig = plt.figure(figsize=(20, 16), facecolor="#0a0a0a")
        fig.suptitle(
            f"SchizoBrain Clinical Report\n"
            f"Diagnosis: {results['diagnosis']} | "
            f"Confidence: {results['confidence']:.1f}% | "
            f"Probability: {results['probability']:.4f}"
            + (f" | Age: {age_value}" if age_value else "")
            + (f" | Gender: {gender_value}" if gender_value else ""),
            color="white", fontsize=14, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(
            4, 3, figure=fig,
            hspace=0.35, wspace=0.1,
            left=0.05, right=0.95,
            top=0.92, bottom=0.05
        )

        row_labels = [
            "Original MRI",
            "Grad-CAM (CNN)",
            "Attention Rollout (ViT)",
            "Combined Heatmap",
        ]
        col_labels = ["Axial", "Coronal", "Sagittal"]

        maps = [
            None,
            results["gradcam"],
            results["attention"],
            results["combined"],
        ]

        for row_idx, (label, heatmap) in enumerate(
            zip(row_labels, maps)
        ):
            for col_idx, (col_label, (slice_fn, slice_idx)) in enumerate(
                zip(col_labels, [
                    (lambda v, s: v[s, :, :], ax_s),
                    (lambda v, s: v[:, s, :], cor_s),
                    (lambda v, s: v[:, :, s], sag_s),
                ])
            ):
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # MRI slice
                mri_slice = slice_fn(mri_np, slice_idx)
                ax.imshow(
                    mri_slice.T, cmap="gray",
                    origin="lower", aspect="equal"
                )

                # Overlay heatmap if applicable
                if heatmap is not None:
                    heat_slice = slice_fn(heatmap, slice_idx)
                    
                    # Mask out low activations (< 0.15) to remove background noise
                    # This makes the highly activated regions stand out much more clearly
                    import numpy as np
                    masked_heat = np.ma.masked_where(heat_slice < 0.15, heat_slice)

                    ax.imshow(
                        masked_heat.T, cmap=brain_cmap,
                        alpha=0.85,  # Increased from 0.6 to make colors "pop" more
                        origin="lower", aspect="equal",
                        vmin=0, vmax=1,
                    )

                # Labels
                if col_idx == 0:
                    ax.set_ylabel(
                        label, color="white",
                        fontsize=9, fontweight="bold"
                    )
                if row_idx == 0:
                    ax.set_title(
                        col_label, color="white",
                        fontsize=10, fontweight="bold"
                    )

                ax.axis("off")
                ax.set_facecolor("#0a0a0a")

        # ── Color bar ─────────────────────────────────────────
        cbar_ax = fig.add_axes([0.96, 0.1, 0.01, 0.8])
        sm = plt.cm.ScalarMappable(cmap=brain_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(
            "Importance", color="white", fontsize=9
        )
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # ── Disclaimer ────────────────────────────────────────
        fig.text(
            0.5, 0.01,
            "⚠️  For clinical decision support only. "
            "Not a replacement for professional diagnosis.",
            ha="center", color="#888888", fontsize=8
        )

        if save_path:
            plt.savefig(
                save_path, dpi=150,
                bbox_inches="tight",
                facecolor="#0a0a0a"
            )
            print(f"📊 Report saved → {save_path}")

        # plt.show()  # Removed to prevent popping up during API request
        plt.close(fig) # Free memory
        return fig

    def visualize_attention_heads(
        self,
        mri: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        save_path: Optional[str] = None,
    ):
        """
        Visualize what each attention head specializes in.

        Shows axial slice overlaid with attention from each
        of the 4 heads in the last encoder block.

        This reveals clinical specialization:
            Head 0: might focus on ventricles
            Head 1: might focus on prefrontal cortex
            Head 2: might focus on temporal lobe
            Head 3: might focus on hippocampus
        """
        device = next(self.model.parameters()).device
        mri    = mri.to(device)
        age    = age.to(device)
        gender = gender.to(device)

        head_maps = self.attention.get_per_head_attention(
            mri, age, gender
        )
        mri_np = mri.squeeze().cpu().numpy()
        mid = mri_np.shape[0] // 2

        colors = [
            (0, 0, 1, 0), (0, 0, 1, 0.3),
            (0, 1, 0, 0.5), (1, 1, 0, 0.7), (1, 0, 0, 0.9),
        ]
        brain_cmap = LinearSegmentedColormap.from_list(
            "brain_heat", colors, N=256
        )

        fig, axes = plt.subplots(
            1, len(head_maps), figsize=(5 * len(head_maps), 5),
            facecolor="#0a0a0a"
        )
        fig.suptitle(
            "Attention Head Specialization (Last ViT Block)",
            color="white", fontsize=13, fontweight="bold"
        )

        for i, (ax, head_map) in enumerate(zip(axes, head_maps)):
            ax.imshow(
                mri_np[mid].T, cmap="gray",
                origin="lower", aspect="equal"
            )
            ax.imshow(
                head_map[mid].T, cmap=brain_cmap,
                alpha=0.65, origin="lower",
                aspect="equal", vmin=0, vmax=1,
            )
            ax.set_title(
                f"Head {i+1}", color="white",
                fontsize=10, fontweight="bold"
            )
            ax.axis("off")
            ax.set_facecolor("#0a0a0a")

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path, dpi=150,
                bbox_inches="tight",
                facecolor="#0a0a0a"
            )
            print(f"📊 Head attention saved → {save_path}")

        plt.show()
        return fig

    def cleanup(self):
        """Remove all hooks — always call when done."""
        self.gradcam.remove_hooks()
        self.attention.remove_hooks()
        print("🧹 Hooks removed")


# ==============================================================
# QUICK TEST
# ==============================================================

if __name__ == "__main__":

    import sys
    sys.path.append(".")
    from hybrid_model_v3 import SchizoBrain, AgeNormalizer

    print("=" * 60)
    print("SchizoBrain Interpretability Test")
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
    ).to(device)
    model.eval()

    # Dummy patient data
    mri    = torch.randn(1, 1, 96, 96, 96).to(device)
    normalizer = AgeNormalizer()
    normalizer.fit([18, 65])
    age    = torch.tensor([[normalizer.transform(28)]]).to(device)
    gender = torch.tensor([[1.0]]).to(device)  # Male

    # Initialize interpreter
    interpreter = SchizoBrainInterpreter(model)

    # Generate explanation
    print("\nGenerating interpretability maps...")
    results = interpreter.explain(mri, age, gender, target_class=1)

    print(f"\nResults:")
    print(f"  Diagnosis:   {results['diagnosis']}")
    print(f"  Probability: {results['probability']:.4f}")
    print(f"  Confidence:  {results['confidence']:.1f}%")
    print(f"  Grad-CAM shape:   {results['gradcam'].shape}")
    print(f"  Attention shape:  {results['attention'].shape}")
    print(f"  Combined shape:   {results['combined'].shape}")
    print(f"  Grad-CAM range:   [{results['gradcam'].min():.3f}, "
          f"{results['gradcam'].max():.3f}]")
    print(f"  Attention range:  [{results['attention'].min():.3f}, "
          f"{results['attention'].max():.3f}]")

    # Visualize (saves to file in test)
    os.makedirs("experiments", exist_ok=True)
    interpreter.visualize(
        mri, results,
        age_value=28,
        gender_value="Male",
        save_path="experiments/gradcam_report.png",
    )

    # Cleanup
    interpreter.cleanup()

    print("\n✅ Interpretability working correctly!")
    print("   Report saved → experiments/gradcam_report.png")
    print("=" * 60)