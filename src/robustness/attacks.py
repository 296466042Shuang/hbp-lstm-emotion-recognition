"""Simple white-box perturbation utilities for robustness analysis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm_pose_attack(
    model,
    pose: torch.Tensor,
    high_level: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Return an FGSM-perturbed pose tensor for evaluation.

    This helper is intended for robustness testing of the public research model.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    was_training = model.training
    model.eval()

    adv = pose.detach().clone().requires_grad_(True)
    output = model(adv, high_level, sample=False)
    loss = F.cross_entropy(output["logits"], labels.long())
    loss.backward()

    perturbed = (adv + epsilon * adv.grad.sign()).detach()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    return perturbed
