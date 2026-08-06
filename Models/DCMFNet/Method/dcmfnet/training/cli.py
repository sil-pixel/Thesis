"""Command-line entry point for confidential, offline DCMFNet training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from ..artifact import build_model
from .config import TrainingConfig
from .data import (
    GROUP_COLUMN,
    NUM_FUSION_MODALITIES,
    TARGET_COLUMNS,
    create_loader,
    fit_schema,
    grouped_split,
)
from .engine import evaluate, train_model
from .export import save_artifact, save_torchscript


DEFAULT_HYPERPARAMETERS = Path(__file__).resolve().parents[3] / "hyperparameters.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Confidential CATSS CSV")
    parser.add_argument("--target", required=True, choices=sorted(TARGET_COLUMNS))
    parser.add_argument("--output", required=True, type=Path, help="Deployable .pt file")
    parser.add_argument("--hyperparameters", type=Path, default=DEFAULT_HYPERPARAMETERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--torchscript", type=Path)
    return parser


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path, target: str) -> TrainingConfig:
    configurations = json.loads(path.read_text(encoding="utf-8"))
    values = configurations.get(target, configurations.get("default"))
    if values is None:
        raise ValueError(f"No hyperparameters found for target {target!r}")
    return TrainingConfig.from_mapping(values)


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = choose_device(args.device)
    config = load_config(args.hyperparameters, args.target)
    target_column = TARGET_COLUMNS[args.target]

    frame = pd.read_csv(args.data)
    if target_column not in frame:
        raise ValueError(f"Target column {target_column!r} is missing")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.loc[frame[target_column].notna()].copy()
    if frame.empty:
        raise ValueError("No rows contain a usable target")

    development, test = grouped_split(frame, args.test_size, args.seed)
    train, validation = grouped_split(
        development, args.validation_size, args.seed + 1
    )
    schema = fit_schema(train)
    loaders = {
        "train": create_loader(
            train, schema, target_column, config.batch_size,
            shuffle=True, seed=args.seed,
        ),
        "validation": create_loader(
            validation, schema, target_column, config.batch_size,
            shuffle=False, seed=args.seed,
        ),
        "test": create_loader(
            test, schema, target_column, config.batch_size,
            shuffle=False, seed=args.seed,
        ),
    }
    model_config = {
        "num_modalities": NUM_FUSION_MODALITIES,
        "num_layers": config.num_layers,
        "feature_sizes": schema.sizes,
        "se_reduction": config.se_reduction,
        "dropout": config.dropout,
        "hidden_dim_min": config.hidden_dim_min,
    }
    result = train_model(
        build_model(model_config), loaders["train"], loaders["validation"], config, device
    )
    test_metrics = evaluate(result.model, loaders["test"], device)
    output = save_artifact(
        args.output,
        result.model,
        model_config=model_config,
        schema=schema,
        target=target_column,
        metrics=test_metrics,
        training_metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "group_column": GROUP_COLUMN,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "best_epoch": result.best_epoch,
            "validation_metrics": result.validation_metrics,
            "hyperparameters": config.to_dict(),
        },
    )
    if args.torchscript:
        save_torchscript(args.torchscript, result.model, schema.sizes)
    print(f"artifact={output}")
    print(f"held_out_test_metrics={json.dumps(test_metrics, sort_keys=True)}")
    return output


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
