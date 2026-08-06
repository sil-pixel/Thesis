"""Loss functions for imbalanced continuous targets."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


BaseLoss = Literal["mse", "huber"]


def _bin_weights(labels: Tensor, n_bins: int, max_weight: float) -> tuple[Tensor, Tensor]:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    edges = torch.linspace(0, 1, n_bins + 1, device=labels.device)
    indices = (torch.bucketize(labels.detach(), edges) - 1).clamp(0, n_bins - 1)
    counts = torch.bincount(indices, minlength=n_bins).float()
    inverse_frequency = 1.0 / (counts + 1.0)
    weights = (inverse_frequency / inverse_frequency.mean()).clamp(max=max_weight)
    return edges, weights


class ImbalancedRegressionLoss(nn.Module):
    """Inverse-frequency weighted focal MSE or Huber regression loss."""

    def __init__(
        self,
        train_labels: Tensor,
        n_bins: int = 10,
        focal_gamma: float = 2.0,
        max_weight: float = 20.0,
        base_loss: BaseLoss = "mse",
        huber_delta: float = 0.1,
    ) -> None:
        super().__init__()
        if base_loss not in {"mse", "huber"}:
            raise ValueError("base_loss must be 'mse' or 'huber'")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.n_bins = n_bins
        self.focal_gamma = focal_gamma
        self.base_loss_type = base_loss
        self.huber_delta = huber_delta
        edges, weights = _bin_weights(train_labels.flatten(), n_bins, max_weight)
        self.register_buffer("bin_edges", edges)
        self.register_buffer("bin_weights", weights)

    def _sample_weights(self, targets: Tensor) -> Tensor:
        indices = (torch.bucketize(targets.detach(), self.bin_edges) - 1).clamp(
            0, self.n_bins - 1
        )
        return self.bin_weights[indices]

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        if self.base_loss_type == "huber":
            element_loss = F.huber_loss(
                predictions, targets, reduction="none", delta=self.huber_delta
            )
        else:
            element_loss = F.mse_loss(predictions, targets, reduction="none")

        with torch.no_grad():
            focal = (torch.abs(predictions - targets) + 1e-6) ** self.focal_gamma
            focal = focal / (focal.mean() + 1e-6)
        weights = focal * self._sample_weights(targets)
        weights = weights / (weights.mean() + 1e-6)
        return (element_loss * weights).mean()


class InverseFrequencyMSELoss(nn.Module):
    """Inverse-frequency weighted mean-squared-error loss."""

    def __init__(
        self, train_labels: Tensor, n_bins: int = 10, max_weight: float = 20.0
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        edges, weights = _bin_weights(train_labels.flatten(), n_bins, max_weight)
        self.register_buffer("bin_edges", edges)
        self.register_buffer("bin_weights", weights)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        loss = F.mse_loss(predictions, targets, reduction="none")
        indices = (torch.bucketize(targets.detach(), self.bin_edges) - 1).clamp(
            0, self.n_bins - 1
        )
        return (loss * self.bin_weights[indices]).mean()


__all__ = ["ImbalancedRegressionLoss", "InverseFrequencyMSELoss"]
