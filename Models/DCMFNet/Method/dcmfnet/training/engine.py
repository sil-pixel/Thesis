"""Framework-agnostic DCMFNet training loop."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from ..losses import ImbalancedRegressionLoss
from .config import TrainingConfig
from .data import MultiModalDataset


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    best_epoch: int
    validation_metrics: dict[str, float]


def regression_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    rho = 0.0
    if len(targets) > 1:
        target_ranks = _average_ranks(targets)
        prediction_ranks = _average_ranks(predictions)
        if np.std(target_ranks) > 0 and np.std(prediction_ranks) > 0:
            rho = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
    metrics = {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "r2": float(r2_score(targets, predictions)) if len(targets) > 1 else 0.0,
        "rho": rho,
    }
    return {name: value if np.isfinite(value) else 0.0 for name, value in metrics.items()}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values using average ranks for ties, matching Spearman correlation."""
    values = np.asarray(values).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate without mutating the caller-visible model mode."""
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = [value.to(device) for value in inputs]
            predictions.append(model(inputs).reshape(-1).cpu().numpy())
            targets.append(labels.reshape(-1).numpy())
    model.train(was_training)
    return regression_metrics(np.concatenate(targets), np.concatenate(predictions))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> TrainingResult:
    """Train with early stopping and restore the best validation checkpoint."""
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("Training and validation loaders must not be empty")
    model.to(device)
    dataset = train_loader.dataset
    if not isinstance(dataset, MultiModalDataset):
        raise TypeError("train_loader must contain a MultiModalDataset")
    criterion = ImbalancedRegressionLoss(
        dataset.targets,
        n_bins=config.n_bins,
        focal_gamma=config.focal_gamma,
        base_loss=config.base_loss,
        huber_delta=config.huber_delta,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.scheduler_patience,
        factor=config.scheduler_factor,
        min_lr=1e-6,
    )

    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = [value.to(device) for value in inputs]
            labels = labels.to(device).reshape(-1, 1)
            loss = criterion(model(inputs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.detach().item()

        metrics = evaluate(model, validation_loader, device)
        scheduler.step(metrics["rmse"])
        print(
            f"epoch={epoch:03d} loss={running_loss / len(train_loader):.6f} "
            f"val_rmse={metrics['rmse']:.6f} val_mae={metrics['mae']:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        validation_metrics=evaluate(model, validation_loader, device),
    )
