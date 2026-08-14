import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_model import export_targets


class ExportModelTests(unittest.TestCase):
    def test_both_targets_use_their_own_checkpoint_and_output(self) -> None:
        args = argparse.Namespace(
            target="Both",
            checkpoint_pos=Path("artifacts/dcmfnet_pos.pt"),
            checkpoint_neg=Path("artifacts/dcmfnet_neg.pt"),
            output_pos=Path("exports/dcmfnet_pos.pt"),
            output_neg=Path("exports/dcmfnet_neg.pt"),
            metadata_pos=Path("exports/dcmfnet_pos.metadata.json"),
            metadata_neg=Path("exports/dcmfnet_neg.metadata.json"),
        )
        with patch("export_model.export_artifact", side_effect=lambda _, __, output, ___: output) as exporter:
            outputs = export_targets(args)

        self.assertEqual(outputs["Pos"], args.output_pos)
        self.assertEqual(outputs["Neg"], args.output_neg)
        exporter.assert_has_calls(
            [
                call("Pos", args.checkpoint_pos, args.output_pos, args.metadata_pos),
                call("Neg", args.checkpoint_neg, args.output_neg, args.metadata_neg),
            ]
        )


if __name__ == "__main__":
    unittest.main()
