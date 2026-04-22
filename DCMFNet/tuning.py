'''
Hyperparameter tuning for DCMFNet using Optuna.
Author: Silpa Soni Nallacheruvu
Date: 22/04/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

This script imports from training.py and model.py to run multiple training trials with different hyperparameters suggested by Optuna.

Usage:
    pip install optuna plotly
    python tuning.py

Tunes: learning_rate, batch_size, num_epochs, num_layers, weight_decay,
       dropout, base_loss, huber_delta, focal_gamma, n_bins, se_reduction
'''

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
import optuna
from optuna.exceptions import TrialPruned

# Import functions from existing files
from training import (
    random_split, prepare_data, calculate_modality_sizes,
    create_dataloader, evaluate
)
from model import DeepCrossModalFusionModel as DCMFNet
from loss import ImbalancedRegressionLoss


# Fixed
NUM_MODALITIES = 9


def objective(trial, train_df, modality_sizes, model_tag):
    '''
    Single Optuna trial: suggest hyperparameters, train, return best val Spearman rho.
    '''
    # ── Suggest hyperparameters ──
    
    # Training hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8, 16, 32])
    num_epochs = trial.suggest_int('num_epochs', 10, 30)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

    # Model architecture hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 7)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    se_reduction = trial.suggest_int('se_reduction', 2, 8)

    # Loss function hyperparameters
    base_loss = trial.suggest_categorical('base_loss', ['mse', 'huber'])
    huber_delta = trial.suggest_float('huber_delta', 0.01, 0.2)
    focal_gamma = trial.suggest_float('focal_gamma', 0.5, 3.0)
    n_bins = trial.suggest_int('n_bins', 5, 20)

    # ── Create train/val split ──
    seed = 42
    train_split, val_split = random_split(train_df, test_size=0.2, random_state=seed)
    X_train, Y_train = prepare_data(train_split, model_tag)
    X_val, Y_val = prepare_data(val_split, model_tag)
    train_dataloader = create_dataloader(X_train, Y_train, batch_size)
    val_dataloader = create_dataloader(X_val, Y_val, batch_size)

    # Collect training labels for loss initialization
    all_train_labels = []
    for inputs, labels in train_dataloader:
        all_train_labels.append(labels)
    all_train_labels = torch.cat(all_train_labels)

    # ── Initialize model ──
    model = DCMFNet(NUM_MODALITIES, num_layers, modality_sizes, se_reduction=se_reduction, dropout=dropout)

    criterion = ImbalancedRegressionLoss(
        all_train_labels,
        n_bins=n_bins,
        focal_gamma=focal_gamma,
        base_loss=base_loss,
        huber_delta=huber_delta
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ── Training loop (minimal, no printing) ──
    best_val_spearman = float('-inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate using training.py's evaluate function
        val_metrics, _, _ = evaluate(model, val_dataloader)
        val_spearman = val_metrics['spearman_rho']

        # Handle NaN (constant predictions)
        if np.isnan(val_spearman):
            val_spearman = -1.0

        best_val_spearman = max(best_val_spearman, val_spearman)

        # Report to Optuna for pruning
        trial.report(val_spearman, epoch)
        if trial.should_prune():
            raise TrialPruned()

    return best_val_spearman


def print_best_params(study, model_tag):
    '''Print the best hyperparameters in a copy-paste format for training.py'''
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best {model_tag} trial")
    print(f"{'='*60}")
    print(f"  Spearman rho: {best.value:.4f}")
    print(f"\n  Copy these into training.py:")
    print(f"  {'─'*40}")
    for key, value in best.params.items():
        print(f"  {key} = {repr(value)}")
    print(f"  {'─'*40}\n")


def save_visualizations(study, model_tag):
    '''Save Optuna visualization plots as HTML files.'''
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )

        fig = plot_optimization_history(study)
        fig.write_html(f'{model_tag}_optimization_history.html')

        fig = plot_param_importances(study)
        fig.write_html(f'{model_tag}_param_importances.html')

        fig = plot_parallel_coordinate(study)
        fig.write_html(f'{model_tag}_parallel_coordinate.html')

        fig = plot_slice(study)
        fig.write_html(f'{model_tag}_slice_plot.html')

        print(f"Visualizations saved as {model_tag}_*.html")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")
    except Exception as e:
        print(f"Could not generate some visualizations: {e}")


if __name__ == "__main__":
    # ── Load data ──
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape: {df.shape}")
    modality_sizes = calculate_modality_sizes(df)

    # Hold out test set — NOT used during tuning
    train_df, test_df = random_split(df, test_size=0.25, random_state=42)

    # ── Configuration ──
    N_TRIALS = 50           # adjust based on compute budget
    MODEL_TAGS = ["Pos", "Neg"]

    for model_tag in MODEL_TAGS:
        print(f"\n{'='*60}")
        print(f"  Tuning {model_tag} symptom model ({N_TRIALS} trials)")
        print(f"{'='*60}")

        # Create study
        study = optuna.create_study(
            direction='maximize',       # maximize Spearman rho
            study_name=f'DCMFNet_{model_tag}',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,     # run first 5 trials fully
                n_warmup_steps=5        # don't prune before epoch 5
            )
        )

        # Run optimization
        start_time = time.time()
        study.optimize(
            lambda trial: objective(trial, train_df, modality_sizes, model_tag),
            n_trials=N_TRIALS,
            show_progress_bar=True
        )
        elapsed = (time.time() - start_time) / 60

        # ── Results ──
        n_pruned = len([t for t in study.trials 
                        if t.state == optuna.trial.TrialState.PRUNED])
        n_complete = len([t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"\nCompleted in {elapsed:.1f} minutes")
        print(f"Trials: {n_complete} complete, {n_pruned} pruned")

        # Print best params
        print_best_params(study, model_tag)

        # Save all trial results to CSV
        results_df = study.trials_dataframe()
        results_df.to_csv(f'{model_tag}_optuna_results.csv', index=False)
        print(f"All trial results saved to '{model_tag}_optuna_results.csv'")

        # Save visualizations
        save_visualizations(study, model_tag)