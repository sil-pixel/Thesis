"""Confidential-data loading, splitting, and preprocessing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..schema import FeatureSchema


NUM_FUSION_MODALITIES = 9
GROUP_COLUMN = "cmpair"
MODALITY_PATTERNS = (
    "SUD15",
    "PRS",
    "SCZ15",
    "ADHD9",
    "ASD9",
    "ACE15",
    "ACE18",
    "SUD18",
    "SES",
    "SEX",
    r"batch_.*_x_PC",
)
TARGET_COLUMNS = {"Pos": "SCZ18_Pos_Norm", "Neg": "SCZ18_Neg_Norm"}


class MultiModalDataset(Dataset):
    """Tensor-backed dataset for ordered modality arrays."""

    def __init__(self, modalities: Sequence[np.ndarray], targets: np.ndarray) -> None:
        if not modalities:
            raise ValueError("At least one modality is required")
        if any(len(values) != len(targets) for values in modalities):
            raise ValueError("All modalities and targets must have equal row counts")
        self.modalities = [torch.as_tensor(values, dtype=torch.float32) for values in modalities]
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[list[Tensor], Tensor]:
        return [values[index] for values in self.modalities], self.targets[index]


def grouped_split(
    frame: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    group_column: str = GROUP_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete groups between development and holdout frames."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if group_column not in frame:
        raise ValueError(f"Required group column {group_column!r} is missing")
    if frame[group_column].isna().any():
        raise ValueError(f"Group column {group_column!r} contains missing values")
    if frame[group_column].nunique() < 2:
        raise ValueError("At least two distinct groups are required")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=holdout_fraction, random_state=seed
    )
    development, holdout = next(splitter.split(frame, groups=frame[group_column]))
    return frame.iloc[development].copy(), frame.iloc[holdout].copy()


def discover_feature_names(frame: pd.DataFrame) -> list[list[str]]:
    """Discover all expected modalities while preserving CSV column order."""
    groups: list[list[str]] = []
    for index, pattern in enumerate(MODALITY_PATTERNS):
        if index == len(MODALITY_PATTERNS) - 1:
            names = [name for name in frame.columns if re.match(pattern, name)]
        else:
            names = [name for name in frame.columns if name.startswith(pattern)]
        if not names:
            raise ValueError(f"No features found for modality pattern {pattern!r}")
        groups.append(names)

    flat_names = [name for group in groups for name in group]
    duplicates = sorted(name for name, count in Counter(flat_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Features matched multiple modalities: {duplicates}")
    return groups


def fit_schema(frame: pd.DataFrame) -> FeatureSchema:
    """Fit median imputation and standardization on training rows only."""
    feature_names = discover_feature_names(frame)
    medians: list[list[float]] = []
    means: list[list[float]] = []
    scales: list[list[float]] = []
    for names in feature_names:
        values = frame[names].apply(pd.to_numeric, errors="coerce")
        median = values.median(axis=0)
        if median.isna().any():
            empty = median[median.isna()].index.tolist()
            raise ValueError(f"Features contain no training values: {empty}")
        filled = values.fillna(median)
        scale = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        medians.append(median.astype(float).tolist())
        means.append(filled.mean(axis=0).astype(float).tolist())
        scales.append(scale.astype(float).tolist())
    return FeatureSchema(
        modality_names=list(MODALITY_PATTERNS),
        feature_names=feature_names,
        medians=medians,
        means=means,
        scales=scales,
    )


def transform_frame(frame: pd.DataFrame, schema: FeatureSchema) -> list[np.ndarray]:
    """Apply a fitted schema without learning from the supplied frame."""
    missing = sorted(set(schema.flat_feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    transformed: list[np.ndarray] = []
    for names, medians, means, scales in zip(
        schema.feature_names,
        schema.medians,
        schema.means,
        schema.scales,
        strict=True,
    ):
        values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        values = np.where(np.isnan(values), np.asarray(medians, dtype=np.float32), values)
        values = (values - np.asarray(means, dtype=np.float32)) / np.asarray(
            scales, dtype=np.float32
        )
        if not np.isfinite(values).all():
            raise ValueError("Features must be finite numeric values")
        transformed.append(values.astype(np.float32, copy=False))
    return transformed


def create_loader(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    target_column: str,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    targets = frame[target_column].to_numpy(dtype=np.float32)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MultiModalDataset(transform_frame(frame, schema), targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )
