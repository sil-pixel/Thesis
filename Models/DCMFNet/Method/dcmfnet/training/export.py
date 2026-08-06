"""Offline-only exporter for deployable artifacts and private audit metadata."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..artifact import ARTIFACT_VERSION
from ..schema import FeatureSchema


def save_artifact(
    destination: str | Path,
    model: nn.Module,
    *,
    model_config: Mapping[str, Any],
    schema: FeatureSchema,
    target: str,
    metrics: Mapping[str, float],
    training_metadata: Mapping[str, Any],
) -> Path:
    """Write a deployable artifact and a separate offline audit record."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "state_dict": model.cpu().state_dict(),
            "model_config": dict(model_config),
            "feature_schema": schema.to_dict(),
            "target": target,
        },
        output,
    )
    audit = {
        "artifact_version": ARTIFACT_VERSION,
        "target": target,
        "metrics": {name: float(value) for name, value in metrics.items()},
        "training": dict(training_metadata),
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    return output


def save_torchscript(
    destination: str | Path, model: nn.Module, feature_sizes: list[int]
) -> Path:
    """Export weights and graph; preprocessing remains the caller's responsibility."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu().eval()
    example = [torch.zeros(1, size) for size in feature_sizes]
    torch.jit.trace(cpu_model, (example,)).save(str(output))
    return output
