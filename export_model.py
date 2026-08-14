#!/usr/bin/env python3
"""Export positive and negative DCMFNet models for local inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Models/DCMFNet/Method"))

from dcmfnet.artifact import load_artifact


TARGET_NAMES = {
    "Pos": "SCZ18_Pos_Norm",
    "Neg": "SCZ18_Neg_Norm",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["Both", "Pos", "Neg"],
        default="Both",
        help="Export both models by default, or select one target",
    )
    parser.add_argument(
        "--checkpoint-pos",
        "--checkpoint_pos",
        dest="checkpoint_pos",
        type=Path,
        default=ROOT / "artifacts/dcmfnet_pos.pt",
    )
    parser.add_argument(
        "--checkpoint-neg",
        "--checkpoint_neg",
        dest="checkpoint_neg",
        type=Path,
        default=ROOT / "artifacts/dcmfnet_neg.pt",
    )
    parser.add_argument(
        "--output-pos",
        "--output_pos",
        dest="output_pos",
        type=Path,
        default=ROOT / "exports/dcmfnet_pos.pt",
    )
    parser.add_argument(
        "--output-neg",
        "--output_neg",
        dest="output_neg",
        type=Path,
        default=ROOT / "exports/dcmfnet_neg.pt",
    )
    parser.add_argument(
        "--metadata-pos",
        "--metadata_pos",
        dest="metadata_pos",
        type=Path,
        default=ROOT / "exports/dcmfnet_pos.metadata.json",
    )
    parser.add_argument(
        "--metadata-neg",
        "--metadata_neg",
        dest="metadata_neg",
        type=Path,
        default=ROOT / "exports/dcmfnet_neg.metadata.json",
    )
    return parser


def export_artifact(
    target: str,
    checkpoint: Path,
    output: Path,
    metadata_path: Path,
) -> Path:
    """Validate and copy one trained artifact with standalone inference metadata."""
    _, schema, metadata = load_artifact(checkpoint)
    expected_target = TARGET_NAMES[target]
    if metadata["target"] != expected_target:
        raise ValueError(
            f"{checkpoint} contains target {metadata['target']!r}; "
            f"expected {expected_target!r} for {target}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, output)
    metadata_path.write_text(
        json.dumps(
            {
                "target": metadata["target"],
                "model_config": metadata["model_config"],
                "feature_schema": schema.to_dict(),
                "runtime": "dcmfnet",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{target}: artifact={output}")
    print(f"{target}: metadata={metadata_path}")
    return output


def export_targets(args: argparse.Namespace) -> dict[str, Path]:
    """Export the selected model artifacts without sharing target-specific paths."""
    targets = list(TARGET_NAMES) if args.target == "Both" else [args.target]
    outputs: dict[str, Path] = {}
    for target in targets:
        suffix = target.lower()
        outputs[target] = export_artifact(
            target,
            getattr(args, f"checkpoint_{suffix}"),
            getattr(args, f"output_{suffix}"),
            getattr(args, f"metadata_{suffix}"),
        )
    return outputs


def main() -> None:
    export_targets(build_parser().parse_args())


if __name__ == "__main__":
    main()
