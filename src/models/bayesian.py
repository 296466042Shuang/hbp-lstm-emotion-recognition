"""Variational Bayesian layers used by the public HBP-LSTM reference model."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """Mean-field variational linear layer with a standard-normal prior."""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -4.0)

        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -4.0)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def _sample(self, mu: torch.Tensor, rho: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        sigma = self._sigma(rho)
        return mu + sigma * torch.randn_like(mu)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho, sample)
        bias = self._sample(self.bias_mu, self.bias_rho, sample)
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """KL[q(theta)||p(theta)] for a zero-mean isotropic Gaussian prior."""
        prior_var = self.prior_std ** 2

        def kl(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
            sigma = self._sigma(rho)
            var = sigma.square()
            return 0.5 * torch.sum(
                (var + mu.square()) / prior_var
                - 1.0
                + math.log(prior_var)
                - torch.log(var)
            )

        return kl(self.weight_mu, self.weight_rho) + kl(self.bias_mu, self.bias_rho)
