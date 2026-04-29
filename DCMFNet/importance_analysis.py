'''
Importance Analysis from SE Attention Gates in DCMFNet Model.
Extracts gate values from all three SE attention layers and analyzes
which modalities the model weighs most heavily for high-SCZ vs low-SCZ predictions.
Author: Silpa Soni Nallacheruvu
Date: 22/04/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
'''

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import re
from scipy.stats import mannwhitneyu

from train import (
    random_split, prepare_data, calculate_modality_sizes,
    create_dataloader, NUM_MODALITIES, train
)
from model import DeepCrossModalFusionModel as DCMFNet


# Modality labels matching the order in prepare_data
MODALITY_LABELS = [
    "SUD15", "PRS", "SCZ15", "ADHD9", "ASD9",
    "ACE15", "ACE18", "SUD18", "SES", "SEX", "batch*PC"
]

# The first modality (SUD15) is X, the rest are split into
# M=9 fusion modalities + 1 independent modality
FUSION_MODALITY_LABELS = MODALITY_LABELS[1:10]  # PRS through SEX (9 modalities fused with SUD15)
INDEPENDENT_LABEL = MODALITY_LABELS[10]          # batch*PC


'''
Convert num_layers_L to a list of per-modality layer counts.
If already a list, return as-is. If int, expand to [L] * M.
'''
def normalize_layers(num_layers_L, M):
    if isinstance(num_layers_L, int):
        return [num_layers_L] * M
    return list(num_layers_L)


'''
Register forward hooks on all three SE attention layers in the model
to capture gate values during inference.

Returns:
    hooks: list of hook handles (call .remove() to clean up)
    gate_storage: dict that will be populated with gate values
'''
def register_gate_hooks(model):
    gate_storage = {
        'attn_fused': [],
        'attn_independent': [],
        'attn_final': [],
    }
    
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            # SE gate = output / input (since output = input * gates)
            # But safer to capture gates directly from the excitation network
            with torch.no_grad():
                x = input[0]
                gates = module.excitation(x)
                gate_storage[name].append(gates.cpu().numpy())
        return hook_fn
    
    hooks.append(model.attn_fused.register_forward_hook(make_hook('attn_fused')))
    hooks.append(model.attn_independent.register_forward_hook(make_hook('attn_independent')))
    hooks.append(model.attn_final.register_forward_hook(make_hook('attn_final')))
    
    return hooks, gate_storage


'''
Run inference on the dataloader and extract gate values + predictions + targets.
'''
def extract_gates(model, dataloader):
    hooks, gate_storage = register_gate_hooks(model)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Concatenate
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    for key in gate_storage:
        gate_storage[key] = np.concatenate(gate_storage[key], axis=0)
    
    return gate_storage, all_predictions, all_targets


'''
Map gate values back to modality-level averages.

Args:
    layers_per_modality: list of int, one per fusion modality.

For attn_fused: gate dim = sum(layers_per_modality[m] * modality_sizes[m+1])
For attn_independent: gate dim = sum(all modality sizes) - unaffected by layers
For attn_final: fused_dim + independent_dim
'''
def map_gates_to_modalities(gate_values, modality_sizes, layer_name, layers_per_modality):
    M = len(FUSION_MODALITY_LABELS)
    fusion_sizes = modality_sizes[1:M+1]  # sizes of the 9 fusion modalities
    x_size = modality_sizes[0]            # SUD15 size
    ind_size = modality_sizes[-1]         # batch*PC size
    
    if layer_name == 'attn_fused':
        modality_gates = {}
        offset = 0
        for i, (label, size) in enumerate(zip(FUSION_MODALITY_LABELS, fusion_sizes)):
            # Each modality's chunk depends on its own layer count
            chunk_size = layers_per_modality[i] * size
            chunk = gate_values[:, offset:offset + chunk_size]
            modality_gates[label] = chunk.mean(axis=1)  # average gate per sample
            offset += chunk_size
        return modality_gates
    
    elif layer_name == 'attn_independent':
        # X + all modalities including independent
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
        fused_dim = sum(
            layers_per_modality[m] * fusion_sizes[m] for m in range(M)
        )
        independent_dim = x_size + sum(fusion_sizes) + ind_size
        
        fused_gates = gate_values[:, :fused_dim].mean(axis=1)
        independent_gates = gate_values[:, fused_dim:].mean(axis=1)
        return {
            'Fused stream': fused_gates,
            'Independent stream': independent_gates,
        }


'''
Compare gate values between high-SCZ and low-SCZ samples.
Uses the actual target values to define high vs low.
'''
def analyze_high_vs_low(modality_gates, targets, predictions, threshold_percentile=75):
    threshold = np.percentile(targets, threshold_percentile)
    high_mask = targets >= threshold
    low_mask = targets < np.percentile(targets, 100 - threshold_percentile)
    
    results = []
    for modality, gates in modality_gates.items():
        high_gates = gates[high_mask]
        low_gates = gates[low_mask]
        
        # Mann-Whitney U test for significance
        if len(high_gates) > 1 and len(low_gates) > 1:
            stat, p_value = mannwhitneyu(high_gates, low_gates, alternative='two-sided')
        else:
            p_value = float('nan')
        
        results.append({
            'Modality': modality,
            'High SCZ gate (mean)': high_gates.mean(),
            'High SCZ gate (std)': high_gates.std(),
            'Low SCZ gate (mean)': low_gates.mean(),
            'Low SCZ gate (std)': low_gates.std(),
            'Difference': high_gates.mean() - low_gates.mean(),
            'p_value': p_value,
        })
    
    results_df = pd.DataFrame(results).sort_values('Difference', ascending=False)
    return results_df, high_mask, low_mask

'''
Bar plot comparing gate values for high vs low SCZ.
'''
def plot_gate_comparison(results_df, model_tag, layer_name, seed):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modalities = results_df['Modality'].values
    x = np.arange(len(modalities))
    width = 0.35

    ax.bar(x - width/2, results_df['High SCZ gate (mean)'], width,
           yerr=results_df['High SCZ gate (std)'],
           label='High SCZ', color='#e74c3c', alpha=0.8, capsize=3)
    ax.bar(x + width/2, results_df['Low SCZ gate (mean)'], width,
           yerr=results_df['Low SCZ gate (std)'],
           label='Low SCZ', color='#3498db', alpha=0.8, capsize=3)

    for i, row in enumerate(results_df.itertuples()):
        if row.p_value < 0.001:
            ax.text(i, max(row._2, row._4) + row._3 + 0.02, '***', 
                    ha='center', fontsize=10)
        elif row.p_value < 0.01:
            ax.text(i, max(row._2, row._4) + row._3 + 0.02, '**', 
                    ha='center', fontsize=10)
        elif row.p_value < 0.05:
            ax.text(i, max(row._2, row._4) + row._3 + 0.02, '*', 
                    ha='center', fontsize=10)
    
    ax.set_xlabel('Modality')
    ax.set_ylabel('Mean Gate Value')
    ax.set_title(f'{model_tag} - SE Gate Values: High vs Low SCZ ({layer_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{model_tag}_{layer_name}_gate_comparison_seed_{seed}.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_{layer_name}_gate_comparison_seed_{seed}.png'")


'''
Heatmap of average gate values per modality, sorted by target value.
Shows how the model's attention shifts across the SCZ severity spectrum.
'''
def plot_gate_heatmap(gate_storage, targets, modality_sizes, layers_per_modality, model_tag, seed):
    # Use the independent layer since it has all modalities
    ind_gates = gate_storage['attn_independent']
    modality_gates = map_gates_to_modalities(
        ind_gates, modality_sizes, 'attn_independent', layers_per_modality
    )
    
    # Sort samples by target value
    sort_idx = np.argsort(targets)
    
    # Build matrix: rows = modalities, columns = sorted samples
    labels = list(modality_gates.keys())
    gate_matrix = np.array([modality_gates[label][sort_idx] for label in labels])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                              gridspec_kw={'height_ratios': [4, 1]})
    
    # Heatmap
    im = axes[0].imshow(gate_matrix, aspect='auto', cmap='RdYlBu_r',
                         vmin=0.3, vmax=0.7)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Samples (sorted by SCZ severity ->)')
    axes[0].set_title(f'{model_tag} - SE Gate Values Across SCZ Severity Spectrum')
    plt.colorbar(im, ax=axes[0], label='Gate Value')
    
    # Target values below
    axes[1].plot(targets[sort_idx], color='black', linewidth=0.5)
    axes[1].set_ylabel('SCZ Score')
    axes[1].set_xlabel('Samples (sorted)')
    axes[1].set_xlim(0, len(targets))
    
    plt.tight_layout()
    plt.savefig(f'{model_tag}_gate_heatmap_seed_{seed}.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_gate_heatmap_seed_{seed}.png'")


if __name__ == "__main__":
    # Load data and hyperparameters
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    modality_sizes = calculate_modality_sizes(df)
    
    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)
    
    seed = 42
    torch.manual_seed(seed)
    train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
    
    for model_tag in ["Pos", "Neg"]:
        print(f"\n{'='*60}")
        print(f"  Gate Analysis - {model_tag} symptom model")
        print(f"{'='*60}")
        
        hyperparams = hyperparameters_json[model_tag]

        # Normalize to per-modality list (handles both int and list)
        layers_per_modality = normalize_layers(hyperparams["num_layers"], NUM_MODALITIES)
        print(f"  Layers per modality: {dict(zip(FUSION_MODALITY_LABELS, layers_per_modality))}")

        # Train model
        (model, *_) = train(train_df, seed, modality_sizes, model_tag, hyperparams)
        
        # Prepare test data
        X_test, Y_test = prepare_data(test_df, model_tag)
        test_dataloader = create_dataloader(X_test, Y_test, hyperparams["batch_size"])
        
        # Extract gates
        gate_storage, predictions, targets = extract_gates(model, test_dataloader)
        
        print(f"\nGate shapes:")
        for key, val in gate_storage.items():
            print(f"  {key}: {val.shape}")
        
        # -- Analyze each SE layer --
        
        # 1. Fused attention: which fusion modalities matter?
        print(f"\n--- Fused Attention Layer ---")
        fused_modality_gates = map_gates_to_modalities(
            gate_storage['attn_fused'], modality_sizes, 'attn_fused', layers_per_modality
        )
        fused_results, _, _ = analyze_high_vs_low(fused_modality_gates, targets, predictions)
        print(fused_results.to_string(index=False))
        fused_results.to_csv(f'{model_tag}_fused_gate_analysis.csv', index=False)
        plot_gate_comparison(fused_results, model_tag, 'attn_fused', seed)
        
        # 2. Independent attention: which raw modalities matter?
        print(f"\n--- Independent Attention Layer ---")
        ind_modality_gates = map_gates_to_modalities(
            gate_storage['attn_independent'], modality_sizes, 'attn_independent', layers_per_modality
        )
        ind_results, _, _ = analyze_high_vs_low(ind_modality_gates, targets, predictions)
        print(ind_results.to_string(index=False))
        ind_results.to_csv(f'{model_tag}_independent_gate_analysis.csv', index=False)
        plot_gate_comparison(ind_results, model_tag, 'attn_independent', seed)
        
        # 3. Final attention: fused vs independent stream importance
        print(f"\n--- Final Attention Layer ---")
        final_gates = map_gates_to_modalities(
            gate_storage['attn_final'], modality_sizes, 'attn_final', layers_per_modality
        )
        final_results, _, _ = analyze_high_vs_low(final_gates, targets, predictions)
        print(final_results.to_string(index=False))
        final_results.to_csv(f'{model_tag}_final_gate_analysis.csv', index=False)
        plot_gate_comparison(final_results, model_tag, 'attn_final', seed)
        
        # 4. Heatmap across severity spectrum
        plot_gate_heatmap(gate_storage, targets, modality_sizes, layers_per_modality, model_tag, seed)
        
        print(f"\nAll gate analysis results saved for {model_tag} model.")