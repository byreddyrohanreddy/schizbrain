import os
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset
from .transforms import AgeNormalizer

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


