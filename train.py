#!/usr/bin/env python3
"""Train DCMFNet locally using the synthetic dataset by default."""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Models/DCMFNet/Method"))
from dcmfnet.training.cli import build_parser, run

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.data = args.data or ROOT / "data/synthetic_dcmfnet.csv"
    args.output = args.output or ROOT / "artifacts/dcmfnet_pos.pt"
    run(args)
