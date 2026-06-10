'''
Importance Analysis from SE Attention Gates in DCMFNet Model.
Extracts gate values from all three SE attention layers and reports
the mean gate value each modality receives across all test samples,
averaged over multiple random seeds.
Author: Silpa Soni Nallacheruvu
Date: 30/05/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

from train import (
    random_split, prepare_data, calculate_modality_sizes,
    create_dataloader, NUM_MODALITIES, train
)
from model import DeepCrossModalFusionModel as DCMFNet


MODALITY_LABELS = [
    "SUD15", "PRS", "SCZ15", "ADHD9", "ASD9",
    "ACE15", "ACE18", "SUD18", "SES", "SEX", "batch*PC"
]

FUSION_MODALITY_LABELS = MODALITY_LABELS[1:10]
INDEPENDENT_LABEL = MODALITY_LABELS[10]

SEEDS = [42, 43, 44, 45, 46]


def normalize_layers(num_layers_L, M):
    if isinstance(num_layers_L, int):
        return [num_layers_L] * M
    return list(num_layers_L)


def register_gate_hooks(model):
    gate_storage = {'attn_fused': [], 'attn_independent': [], 'attn_final': []}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            with torch.no_grad():
                x = input[0]
                gates = module.excitation(x)
                gate_storage[name].append(gates.cpu().numpy())
        return hook_fn

    hooks.append(model.attn_fused.register_forward_hook(make_hook('attn_fused')))
    hooks.append(model.attn_independent.register_forward_hook(make_hook('attn_independent')))
    hooks.append(model.attn_final.register_forward_hook(make_hook('attn_final')))
    return hooks, gate_storage


def extract_gates(model, dataloader):
    hooks, gate_storage = register_gate_hooks(model)
    model.eval()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

    for h in hooks:
        h.remove()

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    for key in gate_storage:
        gate_storage[key] = np.concatenate(gate_storage[key], axis=0)

    return gate_storage, all_predictions, all_targets


def map_gates_to_modalities(gate_values, modality_sizes, layer_name, layers_per_modality):
    M = len(FUSION_MODALITY_LABELS)
    fusion_sizes = modality_sizes[1:M+1]
    x_size = modality_sizes[0]
    ind_size = modality_sizes[-1]

    if layer_name == 'attn_fused':
        modality_gates = {}
        offset = 0
        for i, (label, size) in enumerate(zip(FUSION_MODALITY_LABELS, fusion_sizes)):
            chunk_size = layers_per_modality[i] * size
            chunk = gate_values[:, offset:offset + chunk_size]
            modality_gates[label] = chunk.mean(axis=1)
            offset += chunk_size
        return modality_gates

    elif layer_name == 'attn_independent':
        all_labels = [MODALITY_LABELS[0]] + FUSION_MODALITY_LABELS + [INDEPENDENT_LABEL]
        all_sizes = [x_size] + fusion_sizes + [ind_size]
        modality_gates = {}
        offset = 0
        for label, size in zip(all_labels, all_sizes):
            chunk = gate_values[:, offset:offset + size]
            modality_gates[label] = chunk.mean(axis=1)
            offset += size
        return modality_gates

    elif layer_name == 'attn_final':
        fused_dim = sum(layers_per_modality[m] * fusion_sizes[m] for m in range(M))
        fused_gates = gate_values[:, :fused_dim].mean(axis=1)
        independent_gates = gate_values[:, fused_dim:].mean(axis=1)
        return {'Fused stream': fused_gates, 'Independent stream': independent_gates}


'''
Bar plot of mean gate value per modality, averaged across seeds.
Error bars = across-seed standard deviation (stability of the estimate).
Y-axis is auto-zoomed to the data range so small differences are visible.
'''
def plot_mean_gates(results_df, model_tag, layer_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    modalities = results_df['Modality'].values
    x = np.arange(len(modalities))
    means = results_df['Mean gate'].values
    errs = results_df['Across-seed std'].values

    bars = ax.bar(x, means, yerr=errs, color='#3498db', alpha=0.85, capsize=3)

    # Zoom y-axis to the data range (including error bars), with padding
    lo = (means - errs).min()
    hi = (means + errs).max()
    pad = (hi - lo) * 0.15 if hi > lo else 0.01
    ymin, ymax = lo - pad, hi + pad
    ax.set_ylim(ymin, ymax)

    # Neutral line only if it falls within the zoomed range
    if ymin <= 0.5 <= ymax:
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Neutral (0.5)')
        ax.legend()

    # Value labels above each bar (above the error cap)
    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + e + pad * 0.25, f'{m:.4f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Modality')
    ax.set_ylabel('Mean Gate Value (mean ± std over 5 seeds)')
    ax.set_title(f'{model_tag} - Mean SE Gate Value per Modality ({layer_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{model_tag}_{layer_name}_mean_gates_5seeds.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_{layer_name}_mean_gates_5seeds.png'")

if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    modality_sizes = calculate_modality_sizes(df)

    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    for model_tag in ["Pos", "Neg"]:
        print(f"\n{'='*60}")
        print(f"  Gate Analysis - {model_tag} symptom model (5 seeds)")
        print(f"{'='*60}")

        hyperparams = hyperparameters_json[model_tag]
        layers_per_modality = normalize_layers(hyperparams["num_layers"], NUM_MODALITIES)
        print(f"  Layers per modality: {dict(zip(FUSION_MODALITY_LABELS, layers_per_modality))}")

        # Collect per-seed mean gates: {layer_name: {modality: [seed1_mean, seed2_mean, ...]}}
        per_seed = {ln: {} for ln in ['attn_fused', 'attn_independent', 'attn_final']}

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            torch.manual_seed(seed)
            train_df, test_df = random_split(df, test_size=0.25, random_state=seed)

            (model, *_) = train(train_df, seed, modality_sizes, model_tag, hyperparams)

            X_test, Y_test = prepare_data(test_df, model_tag)
            test_dataloader = create_dataloader(X_test, Y_test, hyperparams["batch_size"])

            gate_storage, predictions, targets = extract_gates(model, test_dataloader)

            for layer_name in ['attn_fused', 'attn_independent', 'attn_final']:
                modality_gates = map_gates_to_modalities(
                    gate_storage[layer_name], modality_sizes, layer_name, layers_per_modality
                )
                # mean across all samples for this seed
                for modality, gates in modality_gates.items():
                    per_seed[layer_name].setdefault(modality, []).append(gates.mean())

        # Aggregate across seeds
        for layer_name in ['attn_fused', 'attn_independent', 'attn_final']:
            rows = []
            for modality, seed_means in per_seed[layer_name].items():
                seed_means = np.array(seed_means)
                rows.append({
                    'Modality': modality,
                    'Mean gate': seed_means.mean(),
                    'Across-seed std': seed_means.std(),
                    'N seeds': len(seed_means),
                })
            results = pd.DataFrame(rows).sort_values('Mean gate', ascending=False)

            print(f"\n--- {layer_name} (mean gate per modality, averaged over {len(SEEDS)} seeds) ---")
            print(results.to_string(index=False))
            results.to_csv(f'{model_tag}_{layer_name}_mean_gates_5seeds.csv', index=False)
            plot_mean_gates(results, model_tag, layer_name)

        print(f"\nAll 5-seed gate analysis results saved for {model_tag} model.")