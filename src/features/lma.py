"""Helpers for 33-D LMA / kinematic feature sequences."""

from __future__ import annotations

import torch

LMA_DIM = 33


def validate_lma_sequence(features: torch.Tensor) -> torch.Tensor:
    """Validate a tensor shaped [B, T, 33]."""
    if features.ndim != 3 or features.shape[-1] != LMA_DIM:
        raise ValueError(f"expected LMA features with shape [B, T, {LMA_DIM}]")
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def standardize_lma_sequence(
    features: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standardise each feature dimension across batch and time."""
    features = validate_lma_sequence(features)

    if mean is None:
        mean = features.mean(dim=(0, 1), keepdim=True)
    if std is None:
        std = features.std(dim=(0, 1), keepdim=True, unbiased=False)

    return (features - mean) / std.clamp_min(eps)
