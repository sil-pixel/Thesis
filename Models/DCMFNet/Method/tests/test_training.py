import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from dcmfnet.artifact import build_model
from dcmfnet.training.config import TrainingConfig
from dcmfnet.training.data import create_loader, fit_schema, grouped_split
from dcmfnet.training.engine import regression_metrics, train_model
from dcmfnet.training.cli import run_targets


PREFIXES = [
    "SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18",
    "SUD18", "SES", "SEX",
]


class TrainingTests(unittest.TestCase):
    def test_spearman_metric_handles_ties_without_scipy(self) -> None:
        metrics = regression_metrics(
            np.array([1.0, 1.0, 2.0, 3.0]),
            np.array([10.0, 10.0, 20.0, 30.0]),
        )
        self.assertAlmostEqual(metrics["rho"], 1.0)

    def test_both_targets_use_distinct_artifact_paths(self) -> None:
        args = argparse.Namespace(
            target="Both",
            output=Path("artifacts/dcmfnet.pt"),
            torchscript=None,
        )
        with patch("dcmfnet.training.cli.run", side_effect=lambda item: item.output):
            outputs = run_targets(args)
        self.assertEqual(outputs["Pos"], Path("artifacts/dcmfnet_pos.pt"))
        self.assertEqual(outputs["Neg"], Path("artifacts/dcmfnet_neg.pt"))
        self.assertNotEqual(outputs["Pos"], outputs["Neg"])

    def test_grouped_split_has_no_group_leakage(self) -> None:
        frame = pd.DataFrame({"cmpair": np.repeat(np.arange(10), 2)})
        development, holdout = grouped_split(frame, 0.2, seed=42)
        self.assertTrue(set(development.cmpair).isdisjoint(set(holdout.cmpair)))

    def test_one_epoch_training_smoke(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        for group in range(12):
            for _ in range(2):
                row = {"cmpair": group, "SCZ18_Pos_Norm": float(rng.uniform())}
                row.update({f"{prefix}_x": float(rng.normal()) for prefix in PREFIXES})
                row["batch_demo_x_PC"] = float(rng.normal())
                rows.append(row)
        frame = pd.DataFrame(rows)
        train, validation = grouped_split(frame, 0.25, seed=42)
        schema = fit_schema(train)
        train_loader = create_loader(
            train, schema, "SCZ18_Pos_Norm", 8, shuffle=True, seed=42
        )
        validation_loader = create_loader(
            validation, schema, "SCZ18_Pos_Norm", 8, shuffle=False, seed=42
        )
        config = TrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            num_epochs=1,
            weight_decay=0.0,
            num_layers=1,
            dropout=0.0,
            se_reduction=2,
            hidden_dim_min=2,
            base_loss="huber",
            huber_delta=0.1,
            focal_gamma=1.0,
            n_bins=4,
        )
        model = build_model(
            {
                "num_modalities": 9,
                "num_layers": 1,
                "feature_sizes": schema.sizes,
                "se_reduction": 2,
                "dropout": 0.0,
                "hidden_dim_min": 2,
            }
        )
        result = train_model(
            model, train_loader, validation_loader, config, torch.device("cpu")
        )
        self.assertEqual(result.best_epoch, 1)
        self.assertIn("rmse", result.validation_metrics)


if __name__ == "__main__":
    unittest.main()
