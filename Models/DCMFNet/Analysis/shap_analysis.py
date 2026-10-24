'''
SHAP and grouped permutation importance for DCMFNet, compared with the SE attention gates.
Checks whether the gate values reported by importance_analysis_avg.py agree with two model-agnostic
attribution methods computed on the same models and test sets:
    1. SHAP (expected gradients, shap.GradientExplainer), summed per modality.
    2. Grouped permutation importance: drop in test R² when all items of one modality are shuffled together.
Gates, SHAP and permutation come from the same retrained model per seed, so they describe the same model.
Date: 23/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from the folder containing catss_final_data.csv and hyperparameters.json
(the same set-up as importance_analysis_avg.py). Requires: pip install shap
    python shap_analysis.py

Output (aggregated over test rows; no participant-level values are saved):
    DCMFNet_attribution_per_seed.csv        : importance per modality, source and seed (input to attribution_agreement.py)
    {tag}_shap_feature_importance_5seeds.csv: mean |SHAP| per input feature, mean and SD across seeds
    {tag}_shap_modality_importance_5seeds.png
    DCMFNet_retrained_test_metrics.csv      : test R² of the retrained models, to compare with the final run

The attributions describe what the trained model relies on for its predictions. They are associations
within the model, not causal effects of the predictors.
'''

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The legacy training code and the gate helpers are imported from the repository folders
ANALYSIS_DIR = Path(__file__).resolve().parent
METHOD_DIR = ANALYSIS_DIR.parent / "Method"
for folder in [METHOD_DIR, ANALYSIS_DIR]:
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from train import (
    random_split, prepare_data, calculate_modality_sizes, create_dataloader, NUM_MODALITIES, train
)
from dcmfnet_test_predictions import train_seed_models
from importance_analysis_avg import (
    MODALITY_LABELS, SEEDS, normalize_layers, extract_gates, map_gates_to_modalities
)


N_BACKGROUND = 100        # training rows used as the SHAP reference distribution
N_SHAP_SAMPLES = 200      # expected-gradients samples per explained row
N_PERMUTATION_REPEATS = 10

# Column prefixes in the same order as prepare_data(), used to name the input features
MODALITY_PATTERNS = ["^SUD15", "^PRS", "^SCZ15", "^ADHD9", "^ASD9", "^ACE15", "^ACE18",
                     "^SUD18", "^SES", "^SEX", "^batch_.*_x_PC"]


'''
shap.GradientExplainer calls model(*inputs); DCMFNet expects one list of modality tensors.
'''
class ListInputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(list(inputs))


def r2_score_np(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model([torch.tensor(x, dtype=torch.float32) for x in X]).numpy().flatten()


def feature_names(df):
    return [list(df.filter(regex=pattern).columns) for pattern in MODALITY_PATTERNS]


'''
SHAP values per input feature for every test row (expected gradients against a training background).
Returns one array per modality, shape (n_test, n_features_in_modality).
'''
def compute_shap(model, X_train, X_test, seed):
    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(X_train[0]), size=min(N_BACKGROUND, len(X_train[0])), replace=False)
    background = [torch.tensor(x[background_idx], dtype=torch.float32) for x in X_train]
    test_inputs = [torch.tensor(x, dtype=torch.float32) for x in X_test]

    wrapper = ListInputWrapper(model).eval()
    explainer = shap.GradientExplainer(wrapper, background)
    shap_values = explainer.shap_values(test_inputs, nsamples=N_SHAP_SAMPLES, rseed=seed)

    shap_per_modality = []
    for values, x in zip(shap_values, X_test):
        values = np.asarray(values)
        if values.ndim == 3:          # trailing output dimension in newer shap versions
            values = values[..., 0]
        if values.shape != x.shape:
            raise ValueError(f"Unexpected SHAP shape {values.shape} for input of shape {x.shape}")
        shap_per_modality.append(values)

    # Sanity check: expected gradients are approximately additive
    sum_shap = np.sum([v.sum(axis=1) for v in shap_per_modality], axis=0)
    expected_gap = predict(model, X_test) - predict(model, [x[background_idx] for x in X_train]).mean()
    print(f"    SHAP additivity check: mean sum(SHAP) = {sum_shap.mean():.4f}, "
          f"mean f(x) - E[f(background)] = {expected_gap.mean():.4f}")
    return shap_per_modality


'''
Modality importance from SHAP: sum the SHAP values of the modality's items per row, then take mean |.|.
Summing before the absolute value gives the attribution of the modality as a group.
'''
def shap_modality_importance(shap_per_modality):
    return {label: float(np.mean(np.abs(values.sum(axis=1))))
            for label, values in zip(MODALITY_LABELS, shap_per_modality)}


'''
Grouped permutation importance: shuffle all columns of one modality with the same row permutation
(keeping the items of the modality together), and record the drop in test R².
'''
def permutation_modality_importance(model, X_test, Y_test, seed):
    rng = np.random.default_rng(seed)
    baseline_r2 = r2_score_np(Y_test, predict(model, X_test))
    importance = {}
    for m, label in enumerate(MODALITY_LABELS):
        drops = []
        for _ in range(N_PERMUTATION_REPEATS):
            X_perm = [x.copy() for x in X_test]
            X_perm[m] = X_perm[m][rng.permutation(len(X_perm[m]))]
            drops.append(baseline_r2 - r2_score_np(Y_test, predict(model, X_perm)))
        importance[label] = float(np.mean(drops))
    return importance, baseline_r2


def plot_modality_importance(per_seed_df, model_tag):
    sources = ["DCMFNet_gate_independent", "DCMFNet_SHAP", "DCMFNet_permutation"]
    titles = ["Independent-stream gate", "SHAP (mean |sum per modality|)", "Permutation (drop in R²)"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, source, title in zip(axes, sources, titles):
        df = per_seed_df[(per_seed_df['model_tag'] == model_tag) & (per_seed_df['source'] == source)]
        summary = df.groupby('modality')['importance'].agg(['mean', 'std']).reindex(MODALITY_LABELS)
        ax.barh(summary.index, summary['mean'], xerr=summary['std'], color="#6B0F1A", alpha=0.85, capsize=2)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("mean ± SD over 5 seeds")
    fig.suptitle(f"DCMFNet {model_tag} — modality importance by method", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{model_tag}_shap_modality_importance_5seeds.png", dpi=300)
    plt.close()
    print(f"Saved: '{model_tag}_shap_modality_importance_5seeds.png'")


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    modality_sizes = calculate_modality_sizes(df)
    names_per_modality = feature_names(df)

    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    attribution_rows = []
    metric_rows = []
    feature_shap_per_seed = {"Pos": [], "Neg": []}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SHAP / permutation / gate analysis - seed {seed}")
        print(f"{'='*60}")
        train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
        # Both models are trained before any attribution, in the order of the final run, so the attribution
        # steps (which use random numbers) do not change the Neg model
        models = train_seed_models(train_df, test_df, seed, modality_sizes, hyperparameters_json)

        for model_tag in ["Pos", "Neg"]:
            print(f"\n  --- {model_tag} model, seed {seed} ---")
            hyperparams = hyperparameters_json[model_tag]
            layers_per_modality = normalize_layers(hyperparams["num_layers"], NUM_MODALITIES)
            model = models[model_tag]
            model.eval()

            X_train, _ = prepare_data(train_df, model_tag)
            X_test, Y_test = prepare_data(test_df, model_tag)

            # 1. Gates (same extraction as importance_analysis_avg.py)
            test_dataloader = create_dataloader(X_test, Y_test, hyperparams["batch_size"])
            gate_storage, _, _ = extract_gates(model, test_dataloader)
            for layer_name, source in [('attn_independent', 'DCMFNet_gate_independent'),
                                       ('attn_fused', 'DCMFNet_gate_fused')]:
                modality_gates = map_gates_to_modalities(
                    gate_storage[layer_name], modality_sizes, layer_name, layers_per_modality
                )
                for modality, gates in modality_gates.items():
                    attribution_rows.append({'model_tag': model_tag, 'seed': seed, 'source': source,
                                             'modality': modality, 'importance': float(gates.mean())})

            # 2. SHAP
            shap_per_modality = compute_shap(model, X_train, X_test, seed)
            for modality, value in shap_modality_importance(shap_per_modality).items():
                attribution_rows.append({'model_tag': model_tag, 'seed': seed, 'source': 'DCMFNet_SHAP',
                                         'modality': modality, 'importance': value})
            feature_shap_per_seed[model_tag].append(pd.Series(
                np.concatenate([np.abs(v).mean(axis=0) for v in shap_per_modality]),
                index=[name for names in names_per_modality for name in names]
            ))

            # 3. Grouped permutation importance
            permutation, baseline_r2 = permutation_modality_importance(model, X_test, Y_test, seed)
            for modality, value in permutation.items():
                attribution_rows.append({'model_tag': model_tag, 'seed': seed, 'source': 'DCMFNet_permutation',
                                         'modality': modality, 'importance': value})
            metric_rows.append({'model_tag': model_tag, 'seed': seed, 'test_r2': baseline_r2})
            print(f"    Retrained model test R²: {baseline_r2:.4f}")

    # Feature-level mean |SHAP|, summarised across seeds
    for model_tag in ["Pos", "Neg"]:
        feature_df = pd.concat(feature_shap_per_seed[model_tag], axis=1)
        feature_summary = pd.DataFrame({
            'feature': feature_df.index,
            'modality': [label for label, names in zip(MODALITY_LABELS, names_per_modality) for _ in names],
            'mean_abs_shap': feature_df.mean(axis=1).values,
            'across_seed_std': feature_df.std(axis=1, ddof=0).values,
        }).sort_values('mean_abs_shap', ascending=False)
        feature_summary.to_csv(f"{model_tag}_shap_feature_importance_5seeds.csv", index=False)
        print(f"Saved: '{model_tag}_shap_feature_importance_5seeds.csv'")

    attribution_df = pd.DataFrame(attribution_rows)
    attribution_df.to_csv("DCMFNet_attribution_per_seed.csv", index=False)
    print("Saved: 'DCMFNet_attribution_per_seed.csv'")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv("DCMFNet_retrained_test_metrics.csv", index=False)
    print("Saved: 'DCMFNet_retrained_test_metrics.csv'")
    print("Compare these R² values with Final Best Model/{Pos,Neg}_test_results_per_seed.csv: on the same machine and "
          "software they should be identical.")

    for model_tag in ["Pos", "Neg"]:
        plot_modality_importance(attribution_df, model_tag)
