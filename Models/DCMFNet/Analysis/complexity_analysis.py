'''
Interaction Complexity Analysis for DCMFNet — Multi-seed version.
Author: Silpa Soni Nallacheruvu
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
Date: 29/04/2026

Purpose:
    Identify which SUD15 * modality interactions are more complex by varying
    the number of IGF layers per modality independently.
    All results averaged over 5 seeds for robustness.

Usage:
    python complexity_analysis.py
'''

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import copy
import time
import re

from train import (
    random_split, prepare_data, calculate_modality_sizes,
    create_dataloader, create_cross_validation_data_loaders,
    evaluate, NUM_MODALITIES
)
from model import DeepCrossModalFusionModel as DCMFNet
from loss import ImbalancedRegressionLoss


# Fusion modality labels (modalities 1-9 that are fused with SUD15)
FUSION_LABELS = ["PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX"]

SEEDS = [42, 43, 44, 45, 46]
DEPTH_RANGE = range(1, 6)


def train_single_config(train_df, modality_sizes, model_tag, hyperparams, layer_config, seed):
    '''
    Train a model with a specific per-modality layer configuration and seed.
    Returns val metrics.
    '''
    torch.manual_seed(seed)

    hp = dict(hyperparams)
    hp["num_layers"] = layer_config

    train_split, val_split = random_split(train_df, test_size=0.2, random_state=seed)
    X_train, Y_train = prepare_data(train_split, model_tag)
    X_val, Y_val = prepare_data(val_split, model_tag)
    train_dataloader = create_dataloader(X_train, Y_train, hp["batch_size"])
    val_dataloader = create_dataloader(X_val, Y_val, hp["batch_size"])

    all_train_labels = []
    for inputs, labels in train_dataloader:
        all_train_labels.append(labels)
    all_train_labels = torch.cat(all_train_labels)

    model = DCMFNet(
        NUM_MODALITIES, layer_config, modality_sizes,
        se_reduction=hp["se_reduction"], dropout=hp["dropout"],
        hidden_dim_min=hp["hidden_dim_min"]
    )
    criterion = ImbalancedRegressionLoss(
        all_train_labels, n_bins=hp["n_bins"], focal_gamma=hp["focal_gamma"],
        base_loss=hp["base_loss"], huber_delta=hp["huber_delta"]
    )
    optimizer = optim.Adam(model.parameters(), lr=hp["learning_rate"],
                           weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        patience=hp.get("scheduler_patience", 3),
        factor=hp.get("scheduler_factor", 0.5), min_lr=1e-6
    )

    early_stopping_patience = hp.get("early_stopping_patience", 5)
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(hp["num_epochs"]):
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_metrics, _, _ = evaluate(model, val_dataloader)
        val_rmse = val_metrics['rmse']
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_metrics, _, _ = evaluate(model, val_dataloader)
    return val_metrics, model


def train_and_test_config(train_df, test_df, modality_sizes, model_tag, hyperparams, layer_config, seed):
    '''
    Train and evaluate on test set for a given config and seed.
    '''
    val_metrics, model = train_single_config(
        train_df, modality_sizes, model_tag, hyperparams, layer_config, seed
    )
    X_test, Y_test = prepare_data(test_df, model_tag)
    test_dataloader = create_dataloader(X_test, Y_test, hyperparams["batch_size"])
    test_metrics, _, _ = evaluate(model, test_dataloader)
    return val_metrics, test_metrics


def phase1_per_modality_sweep(df, modality_sizes, model_tag, hyperparams, baseline_depth):
    '''
    Phase 1: For each fusion modality, vary its depth from 1 to 5 while
    keeping all others at baseline_depth. Averaged over all seeds.
    '''
    M = NUM_MODALITIES
    results = []

    for mod_idx in range(M):
        mod_name = FUSION_LABELS[mod_idx]
        print(f"\n{'-'*50}")
        print(f"  Sweeping depth for {mod_name} (SUD15 * {mod_name})")
        print(f"{'-'*50}")

        for depth in DEPTH_RANGE:
            layer_config = [baseline_depth] * M
            layer_config[mod_idx] = depth

            seed_metrics = []
            for seed in SEEDS:
                train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
                val_metrics, _ = train_single_config(
                    train_df, modality_sizes, model_tag, hyperparams, layer_config, seed
                )
                seed_metrics.append(val_metrics)

            # Aggregate across seeds
            rmse_vals = [m['rmse'] for m in seed_metrics]
            r2_vals = [m['r2'] for m in seed_metrics]
            spearman_vals = [m['spearman_rho'] for m in seed_metrics]
            pearson_vals = [m['pearson_r'] for m in seed_metrics]

            results.append({
                'modality': mod_name,
                'modality_idx': mod_idx,
                'depth': depth,
                'val_rmse_mean': np.mean(rmse_vals),
                'val_rmse_std': np.std(rmse_vals),
                'val_r2_mean': np.mean(r2_vals),
                'val_r2_std': np.std(r2_vals),
                'val_spearman_mean': np.mean(spearman_vals),
                'val_spearman_std': np.std(spearman_vals),
                'val_pearson_mean': np.mean(pearson_vals),
                'val_pearson_std': np.std(pearson_vals),
            })

            print(f"  {mod_name}={depth}: RMSE={np.mean(rmse_vals):.4f}±{np.std(rmse_vals):.4f}, "
                  f"R²={np.mean(r2_vals):.4f}±{np.std(r2_vals):.4f}, "
                  f"ρ={np.mean(spearman_vals):.4f}±{np.std(spearman_vals):.4f}")

    return pd.DataFrame(results)


def find_optimal_depths(sweep_df):
    '''
    From Phase 1 results, find the depth that minimizes mean val RMSE for each modality.
    '''
    optimal = []
    for mod_name in FUSION_LABELS:
        mod_df = sweep_df[sweep_df['modality'] == mod_name]
        best_row = mod_df.loc[mod_df['val_rmse_mean'].idxmin()]
        optimal.append({
            'modality': mod_name,
            'optimal_depth': int(best_row['depth']),
            'best_rmse_mean': best_row['val_rmse_mean'],
            'best_rmse_std': best_row['val_rmse_std'],
            'best_r2_mean': best_row['val_r2_mean'],
            'best_r2_std': best_row['val_r2_std'],
            'best_spearman_mean': best_row['val_spearman_mean'],
            'best_spearman_std': best_row['val_spearman_std'],
        })
    return pd.DataFrame(optimal)


def phase2_comparison(df, modality_sizes, model_tag, hyperparams, optimal_df, baseline_depth):
    '''
    Phase 2: Compare optimal per-modality config vs uniform baseline,
    averaged over 5 seeds on test set.
    '''
    optimal_layers = optimal_df['optimal_depth'].tolist()
    uniform_layers = [baseline_depth] * NUM_MODALITIES

    print(f"\n{'='*60}")
    print(f"  Phase 2: Optimal vs Uniform ({len(SEEDS)} seeds)")
    print(f"{'='*60}")
    print(f"  Optimal: {dict(zip(FUSION_LABELS, optimal_layers))}")
    print(f"  Uniform: {baseline_depth} for all")

    configs = {
        'Optimal (per-modality)': optimal_layers,
        f'Uniform (L={baseline_depth})': uniform_layers,
    }

    all_results = []
    for config_name, layer_config in configs.items():
        print(f"\n  Training {config_name}...")
        seed_test_metrics = []

        for seed in SEEDS:
            train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
            _, test_metrics = train_and_test_config(
                train_df, test_df, modality_sizes, model_tag, hyperparams, layer_config, seed
            )
            seed_test_metrics.append(test_metrics)
            print(f"    Seed {seed}: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")

        # Aggregate
        metric_names = ['rmse', 'r2', 'spearman_rho', 'pearson_r']
        row = {
            'Config': config_name,
            'Layers': str(dict(zip(FUSION_LABELS, layer_config))) if isinstance(layer_config, list) else str(layer_config),
        }
        for m in metric_names:
            vals = [s[m] for s in seed_test_metrics]
            row[f'test_{m}_mean'] = np.mean(vals)
            row[f'test_{m}_std'] = np.std(vals)

        all_results.append(row)

        print(f"    Mean: RMSE={row['test_rmse_mean']:.4f}±{row['test_rmse_std']:.4f}, "
              f"R²={row['test_r2_mean']:.4f}±{row['test_r2_std']:.4f}")

    return pd.DataFrame(all_results)


def plot_depth_sweep(sweep_df, model_tag):
    '''
    Plot RMSE and R² vs depth for each modality, with error bands.
    '''
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for mod_name in FUSION_LABELS:
        mod_df = sweep_df[sweep_df['modality'] == mod_name].sort_values('depth')
        depths = mod_df['depth'].values

        # RMSE with error band
        rmse_mean = mod_df['val_rmse_mean'].values
        rmse_std = mod_df['val_rmse_std'].values
        line, = axes[0].plot(depths, rmse_mean, marker='o', label=mod_name)
        axes[0].fill_between(depths, rmse_mean - rmse_std, rmse_mean + rmse_std,
                              alpha=0.15, color=line.get_color())

        # R² with error band
        r2_mean = mod_df['val_r2_mean'].values
        r2_std = mod_df['val_r2_std'].values
        line2, = axes[1].plot(depths, r2_mean, marker='o', label=mod_name)
        axes[1].fill_between(depths, r2_mean - r2_std, r2_mean + r2_std,
                              alpha=0.15, color=line2.get_color())

    axes[0].set_ylabel('Validation RMSE (mean ± std)')
    axes[0].set_title(f'{model_tag} — Val RMSE vs IGF Depth per Modality ({len(SEEDS)} seeds)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Number of IGF Layers')
    axes[1].set_ylabel('Validation R² (mean ± std)')
    axes[1].set_title(f'{model_tag} — Val R² vs IGF Depth per Modality ({len(SEEDS)} seeds)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_tag}_depth_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: '{model_tag}_depth_sweep.png'")


def plot_optimal_depths(optimal_df, model_tag, baseline_depth):
    '''
    Bar chart of optimal depth per modality with RMSE std as error bars.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(optimal_df)))

    bars = ax.bar(optimal_df['modality'], optimal_df['optimal_depth'], color=colors,
                  edgecolor='white', linewidth=0.5)

    # RMSE annotation with std
    for bar, row in zip(bars, optimal_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f'RMSE={row.best_rmse_mean:.4f}\n±{row.best_rmse_std:.4f}',
                ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Modality (Interaction with SUD15)')
    ax.set_ylabel('Optimal Number of IGF Layers')
    ax.set_title(f'{model_tag} — Interaction Complexity: Optimal Depth per Modality ({len(SEEDS)} seeds)')
    ax.set_ylim(0, max(optimal_df['optimal_depth']) + 2)
    ax.axhline(y=baseline_depth, color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline_depth})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{model_tag}_optimal_depths.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_optimal_depths.png'")


def plot_comparison(comparison_df, model_tag):
    '''
    Side-by-side bar chart comparing optimal vs uniform on test metrics.
    '''
    metrics = ['test_rmse', 'test_r2', 'test_spearman_rho']
    labels = ['RMSE', 'R²', 'Spearman ρ']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(comparison_df))
    width = 0.5
    colors = ['#00897B', '#3498db']

    for ax, metric, label in zip(axes, metrics, labels):
        means = comparison_df[f'{metric}_mean'].values
        stds = comparison_df[f'{metric}_std'].values
        config_names = comparison_df['Config'].values

        bars = ax.bar(x, means, width, yerr=stds, color=colors,
                      capsize=5, edgecolor='white', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(config_names, fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)

        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(j, m + s + 0.002, f'{m:.4f}±{s:.4f}',
                    ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'{model_tag} — Optimal vs Uniform Test Performance ({len(SEEDS)} seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_tag}_complexity_comparison.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_complexity_comparison.png'")


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    modality_sizes = calculate_modality_sizes(df)

    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    for model_tag in ["Pos", "Neg"]:
        print(f"\n{'='*60}")
        print(f"  Complexity Analysis — {model_tag} symptom model ({len(SEEDS)} seeds)")
        print(f"{'='*60}")

        hyperparams = hyperparameters_json[model_tag]
        baseline_depth = hyperparams["num_layers"]

        # ── Phase 1: Per-modality depth sweep ──
        print(f"\n  PHASE 1: Per-modality depth sweep (baseline={baseline_depth})")
        sweep_df = phase1_per_modality_sweep(
            df, modality_sizes, model_tag, hyperparams, baseline_depth
        )
        sweep_df.to_csv(f'{model_tag}_depth_sweep_results.csv', index=False)
        print(f"\nSweep results saved to '{model_tag}_depth_sweep_results.csv'")

        # Plot sweep
        plot_depth_sweep(sweep_df, model_tag)

        # Find optimal depths
        optimal_df = find_optimal_depths(sweep_df)
        optimal_df.to_csv(f'{model_tag}_optimal_depths.csv', index=False)
        print(f"\nOptimal depths per modality:")
        print(optimal_df.to_string(index=False))

        # Plot optimal depths
        plot_optimal_depths(optimal_df, model_tag, baseline_depth)

        # ── Phase 2: Compare optimal vs uniform on test set ──
        comparison_df = phase2_comparison(
            df, modality_sizes, model_tag, hyperparams, optimal_df, baseline_depth
        )
        comparison_df.to_csv(f'{model_tag}_complexity_comparison.csv', index=False)
        print(f"\n  Comparison (mean ± std across {len(SEEDS)} seeds):")
        print(comparison_df.to_string(index=False))

        # Plot comparison
        plot_comparison(comparison_df, model_tag)

        # ── Interpretation ──
        print(f"\n  Interpretation:")
        optimal_layers = dict(zip(optimal_df['modality'], optimal_df['optimal_depth']))
        deep = {k: v for k, v in optimal_layers.items() if v > baseline_depth}
        shallow = {k: v for k, v in optimal_layers.items() if v < baseline_depth}
        baseline_matches = {k: v for k, v in optimal_layers.items() if v == baseline_depth}

        if deep:
            print(f"    Complex interactions (need more depth): {deep}")
        if shallow:
            print(f"    Simple interactions (need less depth):  {shallow}")
        if baseline_matches:
            print(f"    Moderate interactions (at baseline):    {baseline_matches}")