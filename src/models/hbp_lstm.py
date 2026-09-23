"""Compact Hybrid Bayesian LSTM reference implementation."""

from __future__ import annotations

import torch
from torch import nn

from .bayesian import BayesianLinear


class TemporalBranch(nn.Module):
    """Bidirectional LSTM encoder returning a fixed-length representation."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [B, T, F]")
        encoded, _ = self.lstm(sequence)
        pooled = encoded.mean(dim=1)
        return self.proj(pooled)


class HBPLSTM(nn.Module):
    """Hybrid Bayesian LSTM with low-/high-level temporal feature fusion."""

    def __init__(
        self,
        num_joints: int = 17,
        coord_dim: int = 3,
        high_level_dim: int = 33,
        low_hidden: int = 96,
        high_hidden: int = 64,
        branch_dim: int = 96,
        fusion_dim: int = 128,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.high_level_dim = high_level_dim

        self.low_level = TemporalBranch(
            input_dim=num_joints * coord_dim,
            hidden_dim=low_hidden,
            out_dim=branch_dim,
        )
        self.high_level = TemporalBranch(
            input_dim=high_level_dim,
            hidden_dim=high_hidden,
            out_dim=branch_dim,
        )

        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.bayesian_head = BayesianLinear(fusion_dim, num_classes)

    def forward(
        self,
        pose: torch.Tensor,
        high_level: torch.Tensor,
        sample: bool = True,
    ) -> dict:
        if pose.ndim != 4:
            raise ValueError("pose must have shape [B, T, J, C]")

        b, t, j, c = pose.shape
        if j != self.num_joints or c != self.coord_dim:
            raise ValueError(
                f"expected pose joints/coords {(self.num_joints, self.coord_dim)}, got {(j, c)}"
            )

        if high_level.ndim != 3 or high_level.shape[-1] != self.high_level_dim:
            raise ValueError(
                f"high_level must have shape [B, T, {self.high_level_dim}]"
            )

        low = pose.reshape(b, t, j * c)
        low_feature = self.low_level(low)
        high_feature = self.high_level(high_level)

        fused = self.fusion(torch.cat([low_feature, high_feature], dim=-1))
        logits = self.bayesian_head(fused, sample=sample)

        return {
            "logits": logits,
            "feature": fused,
            "kl": self.bayesian_head.kl_divergence(),
        }

    @torch.no_grad()
    def predict_mc(
        self,
        pose: torch.Tensor,
        high_level: torch.Tensor,
        samples: int = 20,
    ) -> dict:
        """Monte Carlo predictive summary from repeated Bayesian samples."""
        if samples < 2:
            raise ValueError("samples must be >= 2")

        was_training = self.training
        self.eval()

        probabilities = []
        for _ in range(samples):
            logits = self(pose, high_level, sample=True)["logits"]
            probabilities.append(torch.softmax(logits, dim=-1))

        probs = torch.stack(probabilities, dim=0)
        mean_prob = probs.mean(dim=0)
        entropy = -(mean_prob.clamp_min(1e-8) * mean_prob.clamp_min(1e-8).log()).sum(dim=-1)
        variance = probs.var(dim=0, unbiased=False).mean(dim=-1)

        if was_training:
            self.train()

        return {
            "mean_probability": mean_prob,
            "predictive_entropy": entropy,
            "mean_class_variance": variance,
        }
