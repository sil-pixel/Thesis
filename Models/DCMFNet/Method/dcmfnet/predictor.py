"""Validated inference adapter suitable for an agent tool or web API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifact import load_artifact
from .schema import transform_records


DISCLAIMER = (
    "Research decision-support output only. This model is not a diagnosis, has not "
    "been established here as a regulated medical device, and requires independent "
    "clinical review. Do not use it as the sole basis for care decisions."
)


class PredictionError(ValueError):
    """Input cannot be safely transformed for prediction."""


class DCMFNetPredictor:
    def __init__(self, artifact: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.schema, self.metadata = load_artifact(artifact, self.device)

    def schema_response(self) -> dict[str, Any]:
        return {
            "target": self.metadata["target"],
            "modalities": [
                {"name": name, "features": features}
                for name, features in zip(
                    self.schema.modality_names,
                    self.schema.feature_names,
                    strict=True,
                )
            ],
            "disclaimer": DISCLAIMER,
        }

    def predict(self, records: Sequence[Mapping[str, float]]) -> dict[str, Any]:
        try:
            arrays = transform_records(records, self.schema)
        except (TypeError, ValueError) as exc:
            raise PredictionError(str(exc)) from exc
        tensors = [torch.from_numpy(array).to(self.device) for array in arrays]
        with torch.inference_mode():
            scores = self.model(tensors).squeeze(-1).cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError("Model produced a non-finite score")
        return {
            "target": self.metadata["target"],
            "predictions": [
                {"normalized_symptom_severity": float(score)} for score in scores
            ],
            "disclaimer": DISCLAIMER,
        }
