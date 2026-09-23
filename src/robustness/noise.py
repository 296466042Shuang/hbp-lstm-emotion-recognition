"""Generic perturbations for robustness experiments."""

from __future__ import annotations

import torch


def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float = 0.02,
    clamp: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise to an input tensor."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    out = x + sigma * torch.randn_like(x)

    if clamp is not None:
        out = out.clamp(clamp[0], clamp[1])

    return out
