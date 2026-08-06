"""Validated training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

from ..losses import BaseLoss


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int
    weight_decay: float
    num_layers: int | list[int]
    dropout: float
    se_reduction: int
    hidden_dim_min: int
    base_loss: BaseLoss
    huber_delta: float
    focal_gamma: float
    n_bins: int
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0 or self.num_epochs <= 0:
            raise ValueError("batch_size and num_epochs must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.se_reduction <= 0 or self.hidden_dim_min <= 0:
            raise ValueError("Attention dimensions must be positive")
        if self.n_bins < 2 or self.focal_gamma < 0 or self.huber_delta <= 0:
            raise ValueError("Loss parameters are invalid")
        if self.scheduler_patience < 0 or self.early_stopping_patience <= 0:
            raise ValueError("Patience values are invalid")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.base_loss not in {"mse", "huber"}:
            raise ValueError("base_loss must be 'mse' or 'huber'")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be in (0, 1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        return cls(
            learning_rate=float(values["learning_rate"]),
            batch_size=int(values["batch_size"]),
            num_epochs=int(values["num_epochs"]),
            weight_decay=float(values["weight_decay"]),
            num_layers=(
                int(values["num_layers"])
                if isinstance(values["num_layers"], int)
                else [int(depth) for depth in values["num_layers"]]
            ),
            dropout=float(values["dropout"]),
            se_reduction=int(values["se_reduction"]),
            hidden_dim_min=int(values.get("hidden_dim_min", 8)),
            base_loss=cast(BaseLoss, str(values["base_loss"])),
            huber_delta=float(values["huber_delta"]),
            focal_gamma=float(values["focal_gamma"]),
            n_bins=int(values["n_bins"]),
            scheduler_patience=int(values.get("scheduler_patience", 3)),
            scheduler_factor=float(values.get("scheduler_factor", 0.5)),
            early_stopping_patience=int(
                values.get("early_stopping_patience", 5)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
