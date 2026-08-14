import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Models/DCMFNet/Method"))

from generate_synthetic_data import expected_columns, generate_synthetic_data, validate_dataset
from dcmfnet.artifact import build_model
from dcmfnet.training.data import create_loader, fit_schema


class SyntheticDataTests(unittest.TestCase):
    def test_schema_and_validation(self):
        frame = generate_synthetic_data(64, seed=42)
        validate_dataset(frame)
        self.assertEqual(frame.columns.tolist(), expected_columns())
        self.assertEqual(len(frame), 64)

    def test_validation_rejects_unknown_columns(self):
        frame = generate_synthetic_data(16)
        frame["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_dataset(frame)

    def test_loader_batch_and_actual_dcmfnet_forward(self):
        frame = generate_synthetic_data(32, seed=7)
        schema = fit_schema(frame)
        loader = create_loader(frame, schema, "SCZ18_Pos_Norm", 8, shuffle=False, seed=7)
        inputs, targets = next(iter(loader))
        model = build_model({"num_modalities": 9, "num_layers": 1, "feature_sizes": schema.sizes, "se_reduction": 2, "dropout": 0.0, "hidden_dim_min": 2})
        with torch.inference_mode():
            predictions = model(inputs)
        self.assertEqual(len(inputs), 11)
        self.assertEqual(tuple(targets.shape), (8,))
        self.assertEqual(tuple(predictions.shape), (8, 1))
        self.assertTrue(np.isfinite(predictions.numpy()).all())
