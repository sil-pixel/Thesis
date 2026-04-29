'''
Interaction Complexity Analysis for DCMFNet.
Author: Silpa Soni Nallacheruvu
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
Date: 29/04/2026

Purpose:
    Identify which SUD15 × modality interactions are more complex by varying
    the number of IGF layers per modality independently.

Experiment design:
    Phase 1 - Per-modality depth sweep:
        For each of the 9 fusion modalities, vary its IGF depth from 1 to 7
        while keeping all others at a baseline depth (e.g., 3).
        This isolates how much depth each interaction needs.

    Phase 2 - Optimal configuration:
        Combine the best depth for each modality into a single model
        and compare against the uniform-depth baseline.

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

SEED = 42
BASELINE_DEPTH = 3       # default depth for modalities not being varied
DEPTH_RANGE = range(1, 6)  # test depths 1 through 5


def train_single_config(train_df, modality_sizes, model_tag, hyperparams, layer_config):
    '''
    Train a model with a specific per-modality layer configuration.
    Returns val and test metrics.
    '''
    torch.manual_seed(SEED)

    # Override num_layers with the per-modality list
    hp = dict(hyperparams)
    hp["num_layers"] = layer_config

    train_split, val_split = random_split(train_df, test_size=0.2, random_state=SEED)
    X_train, Y_train = prepare_data(train_split, model_tag)
    X_val, Y_val = prepare_data(val_split, model_tag)
    train_dataloader = create_dataloader(X_train, Y_train, hp["batch_size"])
    val_dataloader = create_dataloader(X_val, Y_val, hp["batch_size"])

    # Collect labels for loss
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


def phase1_per_modality_sweep(train_df, modality_sizes, model_tag, hyperparams):
    '''
    Phase 1: For each fusion modality, vary its depth from 1 to 7 while
    keeping all others at BASELINE_DEPTH. Records val RMSE and R2.
    '''
    M = NUM_MODALITIES
    results = []

    for mod_idx in range(M):
        mod_name = FUSION_LABELS[mod_idx]
        print(f"\n{'-'*50}")
        print(f"  Sweeping depth for {mod_name} (SUD15 × {mod_name})")
        print(f"{'-'*50}")

        for depth in DEPTH_RANGE:
            # All modalities at baseline except the one being varied
            layer_config = [BASELINE_DEPTH] * M
            layer_config[mod_idx] = depth

            print(f"  Config: {mod_name}={depth}, others={BASELINE_DEPTH}")
            start = time.time()
            val_metrics, _ = train_single_config(
                train_df, modality_sizes, model_tag, hyperparams, layer_config
            )
            elapsed = time.time() - start

            results.append({
                'modality': mod_name,
                'modality_idx': mod_idx,
                'depth': depth,
                'val_rmse': val_metrics['rmse'],
                'val_r2': val_metrics['r2'],
                'val_spearman': val_metrics['spearman_rho'],
                'val_pearson': val_metrics['pearson_r'],
                'time_seconds': elapsed,
            })

            print(f"    RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}, "
                  f"rho: {val_metrics['spearman_rho']:.4f} ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def find_optimal_depths(sweep_df):
    '''
    From Phase 1 results, find the depth that minimizes val RMSE for each modality.
    '''
    optimal = []
    for mod_name in FUSION_LABELS:
        mod_df = sweep_df[sweep_df['modality'] == mod_name]
        best_row = mod_df.loc[mod_df['val_rmse'].idxmin()]
        optimal.append({
            'modality': mod_name,
            'optimal_depth': int(best_row['depth']),
            'best_rmse': best_row['val_rmse'],
            'best_r2': best_row['val_r2'],
            'best_spearman': best_row['val_spearman'],
        })
    return pd.DataFrame(optimal)


def phase2_optimal_config(train_df, test_df, modality_sizes, model_tag, hyperparams, optimal_df):
    '''
    Phase 2: Train a model using the optimal per-modality depths
    and compare against the uniform baseline.
    '''
    optimal_layers = optimal_df['optimal_depth'].tolist()
    uniform_layers = [BASELINE_DEPTH] * NUM_MODALITIES

    print(f"\n{'='*60}")
    print(f"  Phase 2: Optimal vs Uniform Configuration")
    print(f"{'='*60}")
    print(f"  Optimal layers: {dict(zip(FUSION_LABELS, optimal_layers))}")
    print(f"  Uniform layers: {BASELINE_DEPTH} for all")

    # Train optimal config
    print(f"\n  Training OPTIMAL config...")
    val_metrics_opt, model_opt = train_single_config(
        train_df, modality_sizes, model_tag, hyperparams, optimal_layers
    )

    # Evaluate on test set
    X_test, Y_test = prepare_data(test_df, model_tag)
    test_dataloader = create_dataloader(X_test, Y_test, hyperparams["batch_size"])
    test_metrics_opt, _, _ = evaluate(model_opt, test_dataloader)

    # Train uniform config
    print(f"\n  Training UNIFORM config...")
    val_metrics_uni, model_uni = train_single_config(
        train_df, modality_sizes, model_tag, hyperparams, uniform_layers
    )
    test_metrics_uni, _, _ = evaluate(model_uni, test_dataloader)

    comparison = pd.DataFrame([
        {
            'Config': 'Optimal (per-modality)',
            'Layers': str(dict(zip(FUSION_LABELS, optimal_layers))),
            'Val RMSE': val_metrics_opt['rmse'],
            'Val R2': val_metrics_opt['r2'],
            'Test RMSE': test_metrics_opt['rmse'],
            'Test R2': test_metrics_opt['r2'],
            'Test Spearman': test_metrics_opt['spearman_rho'],
        },
        {
            'Config': f'Uniform (L={BASELINE_DEPTH})',
            'Layers': str(BASELINE_DEPTH),
            'Val RMSE': val_metrics_uni['rmse'],
            'Val R2': val_metrics_uni['r2'],
            'Test RMSE': test_metrics_uni['rmse'],
            'Test R2': test_metrics_uni['r2'],
            'Test Spearman': test_metrics_uni['spearman_rho'],
        },
    ])

    return comparison


def plot_depth_sweep(sweep_df, model_tag):
    '''
    Plot RMSE and R2 vs depth for each modality.
    '''
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for mod_name in FUSION_LABELS:
        mod_df = sweep_df[sweep_df['modality'] == mod_name].sort_values('depth')
        axes[0].plot(mod_df['depth'], mod_df['val_rmse'], marker='o', label=mod_name)
        axes[1].plot(mod_df['depth'], mod_df['val_r2'], marker='o', label=mod_name)

    axes[0].set_ylabel('Validation RMSE')
    axes[0].set_title(f'{model_tag} - Val RMSE vs IGF Depth per Modality')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Number of IGF Layers')
    axes[1].set_ylabel('Validation R2')
    axes[1].set_title(f'{model_tag} - Val R2 vs IGF Depth per Modality')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_tag}_depth_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: '{model_tag}_depth_sweep.png'")


def plot_optimal_depths(optimal_df, model_tag):
    '''
    Bar chart of optimal depth per modality - taller bars = more complex interaction.
    '''
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(optimal_df)))

    bars = ax.bar(optimal_df['modality'], optimal_df['optimal_depth'], color=colors)

    # Add RMSE labels on bars
    for bar, rmse in zip(bars, optimal_df['best_rmse']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'RMSE={rmse:.4f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Modality (Interaction with SUD15)')
    ax.set_ylabel('Optimal Number of IGF Layers')
    ax.set_title(f'{model_tag} - Interaction Complexity: Optimal Depth per Modality')
    ax.set_ylim(0, max(optimal_df['optimal_depth']) + 1.5)
    ax.axhline(y=BASELINE_DEPTH, color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline ({BASELINE_DEPTH})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{model_tag}_optimal_depths.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_optimal_depths.png'")


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    modality_sizes = calculate_modality_sizes(df)

    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    torch.manual_seed(SEED)
    train_df, test_df = random_split(df, test_size=0.25, random_state=SEED)

    for model_tag in ["Pos", "Neg"]:
        print(f"\n{'='*60}")
        print(f"  Complexity Analysis — {model_tag} symptom model")
        print(f"{'='*60}")

        hyperparams = hyperparameters_json[model_tag]

        # -- Phase 1: Per-modality depth sweep --
        print(f"\n  PHASE 1: Per-modality depth sweep (baseline={BASELINE_DEPTH})")
        sweep_df = phase1_per_modality_sweep(
            train_df, modality_sizes, model_tag, hyperparams
        )
        sweep_df.to_csv(f'{model_tag}_depth_sweep_results.csv', index=False)
        print(f"\nSweep results saved to '{model_tag}_depth_sweep_results.csv'")

        # Plot sweep results
        plot_depth_sweep(sweep_df, model_tag)

        # Find optimal depths
        optimal_df = find_optimal_depths(sweep_df)
        optimal_df.to_csv(f'{model_tag}_optimal_depths.csv', index=False)
        print(f"\nOptimal depths per modality:")
        print(optimal_df.to_string(index=False))

        # Plot optimal depths
        plot_optimal_depths(optimal_df, model_tag)

        # -- Phase 2: Compare optimal vs uniform --
        comparison_df = phase2_optimal_config(
            train_df, test_df, modality_sizes, model_tag, hyperparams, optimal_df
        )
        comparison_df.to_csv(f'{model_tag}_complexity_comparison.csv', index=False)
        print(f"\n  Comparison:")
        print(comparison_df.to_string(index=False))

        print(f"\n  Interpretation:")
        optimal_layers = dict(zip(optimal_df['modality'], optimal_df['optimal_depth']))
        deep = {k: v for k, v in optimal_layers.items() if v > BASELINE_DEPTH}
        shallow = {k: v for k, v in optimal_layers.items() if v < BASELINE_DEPTH}
        baseline = {k: v for k, v in optimal_layers.items() if v == BASELINE_DEPTH}

        if deep:
            print(f"    Complex interactions (need more depth): {deep}")
            print(f"      → These SUD15 × modality interactions have non-linear patterns")
            print(f"        that require deeper fusion to capture.")
        if shallow:
            print(f"    Simple interactions (need less depth):  {shallow}")
            print(f"      → These interactions are more linear and don't benefit from")
            print(f"        deep iterative fusion.")
        if baseline:
            print(f"    Moderate interactions (at baseline):    {baseline}")