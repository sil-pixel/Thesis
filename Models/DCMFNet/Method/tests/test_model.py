import unittest

import torch

from dcmfnet.model import DeepCrossModalFusionModel


class ModelTests(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = DeepCrossModalFusionModel(
            M=2,
            L=[1, 2],
            n_features_per_modality=[3, 4, 2, 1],
            dropout=0.0,
            hidden_dim_min=2,
        )
        inputs = [torch.randn(5, size) for size in [3, 4, 2, 1]]
        self.assertEqual(tuple(model(inputs).shape), (5, 1))

    def test_input_count_is_validated(self) -> None:
        model = DeepCrossModalFusionModel(1, 1, [2, 2, 1])
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            model([torch.zeros(1, 2), torch.zeros(1, 2)])

    def test_state_dict_names_remain_artifact_compatible(self) -> None:
        model = DeepCrossModalFusionModel(1, 1, [2, 3, 1])
        keys = set(model.state_dict())
        self.assertIn(
            "igf_modules.0.gated_fusion_layers.0.fusion_layer.W1.weight", keys
        )
        self.assertIn("attn_final.excitation.3.weight", keys)
        self.assertIn("fc.weight", keys)


if __name__ == "__main__":
    unittest.main()
