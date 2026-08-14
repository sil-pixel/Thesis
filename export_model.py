#!/usr/bin/env python3
"""Export a local DCMFNet artifact and inference metadata."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Models/DCMFNet/Method"))
from dcmfnet.artifact import load_artifact

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/dcmfnet_pos.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/dcmfnet_pos.pt")
    args = parser.parse_args()
    _, schema, metadata = load_artifact(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, args.output)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps({"target": metadata["target"], "model_config": metadata["model_config"], "feature_schema": schema.to_dict(), "runtime": "dcmfnet"}, indent=2), encoding="utf-8")
    print(f"artifact={args.output}")
    print(f"metadata={metadata_path}")

if __name__ == "__main__":
    main()
