from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from dcmfnet.artifact import build_model
from dcmfnet.predictor import DCMFNetPredictor, PredictionError
from dcmfnet.training.data import fit_schema
from dcmfnet.training.export import save_artifact


FEATURES = [
    "SUD15_a", "PRS_a", "SCZ15_a", "ADHD9_a", "ASD9_a", "ACE15_a",
    "ACE18_a", "SUD18_a", "SES_a", "SEX_a", "batch_a_x_PC",
]


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{name: float(row + column) for column, name in enumerate(FEATURES)}
         for row in range(4)]
    )


def create_artifact(path: Path) -> DCMFNetPredictor:
    frame = sample_frame()
    schema = fit_schema(frame)
    config = {
        "num_modalities": 9,
        "num_layers": 1,
        "feature_sizes": schema.sizes,
        "se_reduction": 2,
        "dropout": 0.1,
        "hidden_dim_min": 2,
    }
    save_artifact(
        path,
        build_model(config),
        model_config=config,
        schema=schema,
        target="SCZ18_Pos_Norm",
        metrics={"rmse": 0.2},
        training_metadata={"seed": 42},
    )
    return DCMFNetPredictor(path)


class RuntimeTests(unittest.TestCase):
    def test_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            predictor = create_artifact(tmp_path / "model.pt")
            result = predictor.predict([sample_frame().iloc[0].to_dict()])
            self.assertEqual(result["target"], "SCZ18_Pos_Norm")
            self.assertEqual(len(result["predictions"]), 1)
            self.assertIsInstance(
                result["predictions"][0]["normalized_symptom_severity"], float
            )
            self.assertTrue((tmp_path / "model.audit.json").exists())
            payload = torch.load(tmp_path / "model.pt", weights_only=True)
            self.assertNotIn("training", payload)
            self.assertNotIn("metrics", payload)

    def test_missing_feature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = create_artifact(Path(directory) / "model.pt")
            record = sample_frame().iloc[0].to_dict()
            del record["PRS_a"]
            with self.assertRaisesRegex(PredictionError, "PRS_a"):
                predictor.predict([record])

    def test_unknown_feature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictor = create_artifact(Path(directory) / "model.pt")
            record = sample_frame().iloc[0].to_dict()
            record["unexpected"] = 1.0
            with self.assertRaisesRegex(PredictionError, "unexpected"):
                predictor.predict([record])


if __name__ == "__main__":
    unittest.main()
