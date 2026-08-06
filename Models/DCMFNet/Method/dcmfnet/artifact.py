"""Versioned, self-describing DCMFNet artifact format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .model import DeepCrossModalFusionModel
from .schema import FeatureSchema


ARTIFACT_VERSION = 1


def build_model(config: Mapping[str, Any]) -> DeepCrossModalFusionModel:
    return DeepCrossModalFusionModel(
        M=int(config["num_modalities"]),
        L=config["num_layers"],
        n_features_per_modality=list(config["feature_sizes"]),
        se_reduction=int(config["se_reduction"]),
        dropout=float(config["dropout"]),
        hidden_dim_min=int(config["hidden_dim_min"]),
    )


def load_artifact(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DeepCrossModalFusionModel, FeatureSchema, dict[str, Any]]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    payload = torch.load(artifact_path, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Artifact payload must be a dictionary")
    required = {
        "artifact_version",
        "state_dict",
        "model_config",
        "feature_schema",
        "target",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Artifact is missing required fields: {sorted(missing)}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {payload.get('artifact_version')!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    model = build_model(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    schema = FeatureSchema.from_dict(payload["feature_schema"])
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, schema, metadata
