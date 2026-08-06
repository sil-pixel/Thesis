"""Offline-only DCMFNet training utilities.

Nothing in this package is imported by the inference API.
"""

from .config import TrainingConfig
from .engine import TrainingResult, train_model

__all__ = ["TrainingConfig", "TrainingResult", "train_model"]
