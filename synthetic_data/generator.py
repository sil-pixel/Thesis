#!/usr/bin/env python3
"""Generate and validate a fully synthetic DCMFNet-compatible CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DISCLAIMER = (
    "This dataset is fully synthetic and intended only for development, testing, and "
    "demonstration. It does not contain STR participant records and must not be "
    "interpreted as clinically representative data."
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DCMFNET_METHOD = REPO_ROOT / "Models" / "DCMFNet" / "Method"
if str(DCMFNET_METHOD) not in sys.path:
    sys.path.insert(0, str(DCMFNET_METHOD))

GROUP_COLUMN = "cmpair"
TARGET_COLUMNS = ("SCZ18_Pos_Norm", "SCZ18_Neg_Norm")
PRS_FEATURES = [
    "Amygdala_excitatory", "Cerebellar_inhibitory", "CGE_interneuron",
    "Deep_layer_corticothalamic", "Deep_layer_intratelencepha",
    "Eccentric_medium_spiny_neu", "Hippocampal_CA1_3", "Hippocampal_CA4",
    "Hippocampal_dentate_gyrus", "LAMP5_LHX6_and_Chandelier", "Medium_spiny_neuron",
    "MGE_interneuron", "Miscellaneous", "Thalamic_excitatory",
    "Upper_layer_intratelenceph", "all",
]
FEATURES = {
    "SUD15": ["Cigarettes15", "Snuff15", "Alcohol15", "Cannabis15", "OtherDrugs15", "Painkillers_opioids15"],
    "SCZ15": ["PROD_seen_hallucinations9", "Spied15", "Others_Read_thoughts15", "Special_messages15", "Special_powers15", "Under_control_special_power15", "Read_others_minds15", "Seen_hallucinations15", "Extreme_excitement15", "Irritable15", "Unrealistic_abilities15", "Not_tired15", "Too_much_energy15", "Racing_thoughts15", "Talking_fast15", "Sexual_inappropriate15", "Rage_attacks15", "Hear_voices15", "headaches15", "worry15", "unhappy15", "lose_confidence15", "easily_scared15"],
    "ADHD9": [f"var_{i}" for i in range(1, 20)],
    "ASD9": [f"var_{i}" for i in range(1, 18)],
    "ACE15": ["other_bullying15", "bullied_often15", "tease_bullying15", "emotional_bullying15", "rumours_bullying15", "bullying_by_num15", "bullying_time15"],
    "ACE18": ["other_abuse18", "hate_crime18", "emotional_abuse18", "witness_crime18"],
    "SUD18": ["cigarettes18", "snuff18", "alcohol_often18", "drugs_often18"],
    "SES": ["education_father", "birth_country_father", "education_mother", "birth_country_mother"],
}
FEATURES["PRS"] = PRS_FEATURES
FEATURES["SEX"] = ["SEX"]
FEATURES["batch"] = ["batch_1_x_PC1", "batch_2_x_PC1", "batch_1_x_PC2", "batch_2_x_PC2"]
MODALITY_ORDER = ("SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX", "batch")

ORDINAL_RANGES = {"SUD15": (0, 5), "SCZ15": (0, 3), "ADHD9": (0, 2), "ASD9": (0, 2), "ACE15": (1, 6), "SUD18": (0, 7), "SES": (0, 5)}
MISSINGNESS = {"PRS": 0.283, "SCZ15": 0.274, "ADHD9": 0.149, "ASD9": 0.159, "ACE15": 0.046, "ACE18": 0.111, "SUD18": 0.003, "SES": 0.180, "SEX": 0.147}


def expected_columns() -> list[str]:
    names = [(name if prefix in {"batch", "SEX"} else f"{prefix}_{name}") for prefix in MODALITY_ORDER for name in FEATURES[prefix]]
    return [GROUP_COLUMN, *names, *TARGET_COLUMNS]


def _ordinal(rng: np.random.Generator, n: int, lo: int, hi: int, mean: float, missing: float) -> np.ndarray:
    values = np.clip(np.rint(rng.normal(mean, max(0.55, (hi - lo) / 2.7), n)), lo, hi).astype(float)
    values[rng.random(n) < missing] = np.nan
    return values


def generate_synthetic_data(n_samples: int, seed: int = 42) -> pd.DataFrame:
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    latent = rng.normal(size=n_samples)
    for prefix, names in FEATURES.items():
        if prefix in ORDINAL_RANGES:
            lo, hi = ORDINAL_RANGES[prefix]
            mean = {"SUD15": 0.7, "SCZ15": 0.45, "ADHD9": 0.16, "ASD9": 0.07, "ACE15": 1.22, "SUD18": 0.93, "SES": 2.55}[prefix]
            for name in names:
                data[f"{prefix}_{name}"] = _ordinal(rng, n_samples, lo, hi, mean + 0.12 * latent, MISSINGNESS.get(prefix, 0.0))
        elif prefix == "PRS":
            for name in names:
                values = np.clip(rng.normal(-0.085, 0.276, n_samples), -2.648, 0.668)
                data[f"PRS_{name}"] = np.where(rng.random(n_samples) < MISSINGNESS["PRS"], np.nan, values)
        elif prefix == "ACE18":
            for name in names:
                values = (rng.random(n_samples) < 0.887).astype(float)
                values[rng.random(n_samples) < MISSINGNESS[prefix]] = np.nan
                data[f"ACE18_{name}"] = values
        elif prefix == "SEX":
            values = rng.choice([1.0, 2.0], size=n_samples, p=[0.41, 0.59])
            values[rng.random(n_samples) < MISSINGNESS[prefix]] = np.nan
            data["SEX"] = values
        elif prefix == "batch":
            batch = rng.integers(0, 3, n_samples)
            pc1, pc2 = rng.normal(size=(2, n_samples))
            data["batch_1_x_PC1"] = (batch == 1) * pc1
            data["batch_2_x_PC1"] = (batch == 2) * pc1
            data["batch_1_x_PC2"] = (batch == 1) * pc2
            data["batch_2_x_PC2"] = (batch == 2) * pc2
    data[GROUP_COLUMN] = np.repeat(np.arange((n_samples + 1) // 2), 2)[:n_samples]
    risk = 0.15 * np.nan_to_num(data["SUD15_Cannabis15"]) + 0.12 * np.nan_to_num(data["ACE15_other_bullying15"]) + 0.10 * latent + rng.normal(0, 0.10, n_samples)
    data["SCZ18_Pos_Norm"] = np.round(np.clip(0.14 + risk / 4, 0, 0.87) * 46) / 46
    data["SCZ18_Neg_Norm"] = np.round(np.clip(0.26 + risk / 5 + rng.normal(0, 0.06, n_samples), 0, 1) * 33) / 33
    return pd.DataFrame(data, columns=expected_columns())


def validate_dataset(frame: pd.DataFrame) -> None:
    expected = set(expected_columns())
    actual = set(frame.columns)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"Schema mismatch; missing={missing}, unexpected={unexpected}")
    if frame.columns.tolist() != expected_columns():
        raise ValueError("Columns are present but not in the required feature order")
    if frame[GROUP_COLUMN].isna().any() or not pd.api.types.is_integer_dtype(frame[GROUP_COLUMN]):
        raise ValueError("cmpair must be a non-missing integer group identifier")
    for prefix, (lo, hi) in ORDINAL_RANGES.items():
        for name in FEATURES[prefix]:
            values = frame[f"{prefix}_{name}"].dropna()
            if not pd.api.types.is_numeric_dtype(values) or not np.isin(values, np.arange(lo, hi + 1)).all():
                raise ValueError(f"{prefix}_{name} must contain integer values in [{lo}, {hi}]")
    for name in FEATURES["PRS"]:
        values = frame[f"PRS_{name}"].dropna()
        if not pd.api.types.is_float_dtype(frame[f"PRS_{name}"]) or not np.isfinite(values).all() or not values.between(-2.648, 0.668).all():
            raise ValueError(f"PRS_{name} must be finite floating-point data")
    for name in FEATURES["ACE18"]:
        if not set(frame[f"ACE18_{name}"].dropna().unique()).issubset({0.0, 1.0}):
            raise ValueError(f"ACE18_{name} must be binary")
    if not set(frame["SEX"].dropna().unique()).issubset({1.0, 2.0}):
        raise ValueError("SEX must use the 1/2 encoding")
    for target in TARGET_COLUMNS:
        if not pd.api.types.is_float_dtype(frame[target]) or not frame[target].between(0, 1).all():
            raise ValueError(f"{target} must be floating-point values in [0, 1]")
    from dcmfnet.training.data import create_loader, discover_feature_names, fit_schema, transform_frame
    groups = discover_feature_names(frame)
    if [len(group) for group in groups] != [6, 16, 23, 19, 17, 7, 4, 4, 4, 1, 4]:
        raise ValueError("Unexpected modality sizes")
    schema = fit_schema(frame)
    transform_frame(frame, schema)
    for target in TARGET_COLUMNS:
        loader = create_loader(frame, schema, target, batch_size=min(32, len(frame)), shuffle=False, seed=0)
        next(iter(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_dcmfnet.csv"))
    args = parser.parse_args()
    frame = generate_synthetic_data(args.n_samples, args.seed)
    validate_dataset(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote={args.output} rows={len(frame)} columns={len(frame.columns)}")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
