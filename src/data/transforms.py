import torch

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


