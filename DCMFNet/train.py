'''
Training of DCMFNet model
Author: Silpa Soni Nallacheruvu
Date: 18/02/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.


Hyperparameters mentioned in the paper: 
- Learning rate: 1.82e-4
- Batch size: 6
- Number of epochs: 15
- Optimizer: Adam
- Loss function: Sigmoid Cross-entropy loss
- Number of Layers: 5
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DeepCrossModalFusionModel as DCMFNet
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loss import ImbalancedRegressionLoss
import re
import json

# Load hyperparameters json
with open("hyperparameters.json", "r") as f:
    hyperparameters_json = json.load(f)

# Initialize model parameters
'''
9 modalities (input features)
'''
NUM_MODALITIES = 9 

'''
Split the data into test and train sets using twin_id as the identifier and return the train and test dataframes with input and target modalities.
Input:
    df: pandas dataframe containing the data
    test_size: size of the test set
    random_state: random seed for reproducibility
Output:
    train_df: pandas dataframe containing the training data
    test_df: pandas dataframe containing the test data
'''
def random_split(df, test_size=0.25, random_state=42):
    unique_twin_ids = df['cmpair'].unique()
    train_ids, test_ids = train_test_split(unique_twin_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['cmpair'].isin(train_ids)]
    test_df = df[df['cmpair'].isin(test_ids)]
    print(f"Train: {len(train_df)} rows ({len(train_ids)} IDs),"
            f"Test: {len(test_df)} rows ({len(test_ids)} IDs) \n")
    return train_df, test_df


'''
Prepare the dataframe for training by splitting the dataframe into input and target modalities dataframes.
'''
def prepare_data(df, model_tag):
    X = [
        df.filter(regex="^SUD15").to_numpy(),
        df.filter(regex="^PRS").to_numpy(),
        df.filter(regex="^SCZ15").to_numpy(),
        df.filter(regex="^ADHD9").to_numpy(),
        df.filter(regex="^ASD9").to_numpy(),
        df.filter(regex="^ACE15").to_numpy(),
        df.filter(regex="^ACE18").to_numpy(),
        df.filter(regex="^SUD18").to_numpy(),
        df.filter(regex="^SES").to_numpy(),
        df.filter(regex="^SEX").to_numpy(),
        df.filter(regex="^batch_.*_x_PC").to_numpy()
    ]
    if model_tag == "Pos":
        Y = df["SCZ18_Pos_Norm"].to_numpy()
    elif model_tag == "Neg":
        Y = df["SCZ18_Neg_Norm"].to_numpy()

    
    return X, Y

'''
Calculate the modality sizes for the catss dataset
'''
def calculate_modality_sizes(df):
    prefixes = ["SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX"]
    regex_patterns = [r"^batch_.*_x_PC"]
    counts = []
    for prefix in prefixes:
        counts.append(sum(col.startswith(prefix) for col in df.columns))
    for pattern in regex_patterns:
        counts.append(sum(1 for col in df.columns if re.match(pattern, col)))
    
    all_modals = prefixes + regex_patterns
    print(f"modality sizes for {all_modals} : {counts}")
    return counts


'''
Custom dataset class for multi-modal data.
'''
class MultiModalDataset(Dataset):
    def __init__(self, X_list, Y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return [x[idx] for x in self.X], self.Y[idx]


'''
Create a data loader for the given input and target dataframes.
'''
def create_dataloader(X, Y, batch_size):
    dataset = MultiModalDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


'''
Create a cross validation data loader with 20% of the data as the validation set and 80% of the data as the training set.
Input:
        df: pandas dataframe containing the training data
Output:
        train_dataloader: data loader containing the training data
        val_dataloader: data loader containing the validation data
'''
def create_cross_validation_data_loaders(df, seed, model_tag, batch_size):
    # attach the input and target modalities 
    train_df, val_df = random_split(df, test_size=0.2, random_state=seed)
    X_train, Y_train = prepare_data(train_df, model_tag)
    X_val, Y_val = prepare_data(val_df, model_tag)
    train_dataloader = create_dataloader(X_train, Y_train, batch_size)
    val_dataloader = create_dataloader(X_val, Y_val, batch_size)
    return train_dataloader, val_dataloader

'''
Evaluate the model and return regression metrics + raw predictions/targets.
'''
def evaluate(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets = targets.unsqueeze(1)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_predictions = torch.cat(all_predictions).numpy().flatten()
    all_targets = torch.cat(all_targets).numpy().flatten()
    
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    if np.std(all_predictions) < 1e-6:
        print("WARNING: predictions are constant, correlation undefined")
        spearman_rho = float('nan')
        pearson_r = float('nan')
    else:
        spearman_rho, spearman_p_value = spearmanr(all_targets, all_predictions)
        pearson_r, pearson_p_value = pearsonr(all_targets, all_predictions)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p_value,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p_value
    }
    
    print(f"  Pred range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}], "
          f"std: {np.std(all_predictions):.4f}")
    print(f"  RMSE : {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, "
          f"Spearman rho: {spearman_rho:.4f}, Spearman p: {spearman_p_value:.4e}, "
          f"Pearson r: {pearson_r:.4f}, Pearson p: {pearson_p_value:.4e}")
 
    model.train()
    return metrics, all_predictions, all_targets
 

'''
Plot predicted vs actual scatter plot and residual plot.
'''
def plot_predicted_vs_actual(predictions, targets, metrics, seed, model_tag, split_name="val"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: Predicted vs Actual
    ax1 = axes[0]
    ax1.scatter(targets, predictions, alpha=0.4, s=20, edgecolors='none')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    margin = (max_val - min_val) * 0.05
    ax1.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', linewidth=1.5, label='Perfect prediction')
    if np.std(predictions) > 1e-6:
        z = np.polyfit(targets, predictions, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax1.plot(x_line, p_line(x_line), 'b-', alpha=0.7, linewidth=1.5, label='Best fit')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{model_tag} - Predicted vs Actual ({split_name})')
    ax1.legend(loc='upper left')
    textstr = (f"R2 = {metrics['r2']:.4f}\n"
               f"RMSE = {metrics['rmse']:.4f}\n"
               f"Spearman rho = {metrics['spearman_rho']:.4f}\n"
               f"Spearman p = {metrics['spearman_p']:.4e}\n"
               f"Pearson r = {metrics['pearson_r']:.4f}\n"
               f"Pearson p = {metrics['pearson_p']:.4e}")
    ax1.text(0.97, 0.03, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Residual plot
    ax2 = axes[1]
    residuals = targets - predictions
    ax2.scatter(predictions, residuals, alpha=0.4, s=20, edgecolors='none')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title(f'{model_tag} - Residual Plot ({split_name})')
    
    plt.tight_layout()
    plt.savefig(f'{model_tag}_pred_vs_actual_{split_name}_seed_{seed}.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_pred_vs_actual_{split_name}_seed_{seed}.png'")
 
 
'''
Plot training curves: Loss, MAE, Spearman rho, and R2 over epochs.
'''
def plot_training_curves(train_losses, train_rmses, val_rmses, 
                         train_spearmans, val_spearmans,
                         train_r2s, val_r2s, seed, model_tag):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    axes[0, 0].plot(epochs, train_losses, label='Training Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
 
    # MAE
    axes[0, 1].plot(epochs, train_rmses, label='Training RMSE')
    axes[0, 1].plot(epochs, val_rmses, label='Validation RMSE')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Training and Validation RMSE')
    axes[0, 1].legend()
 
    # Spearman rho
    axes[1, 0].plot(epochs, train_spearmans, label='Training Spearman rho')
    axes[1, 0].plot(epochs, val_spearmans, label='Validation Spearman rho')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Spearman rho')
    axes[1, 0].set_title('Training and Validation Spearman rho')
    axes[1, 0].legend()
 
    # R2
    axes[1, 1].plot(epochs, train_r2s, label='Training R2')
    axes[1, 1].plot(epochs, val_r2s, label='Validation R2')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('R2')
    axes[1, 1].set_title('Training and Validation R2')
    axes[1, 1].legend()
    
    plt.suptitle(f'{model_tag} Model - Seed {seed}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{model_tag}_training_curves_seed_{seed}.png', dpi=150)
    plt.close()
    print(f"Saved: '{model_tag}_training_curves_seed_{seed}.png'")
 
 

'''
Train the model on the training data and evaluate the model on the test data.
Input:
    df: pandas dataframe containing the training data
    n_features_per_modality: List of number of features in each modality
    model_tag: tag to distinguihs between positive or negative schizophrenic symptom model
Output:
    model: trained model
    train_losses: list of training losses over epochs
    train_rmses: list of training RMSEs over epochs
    val_rmses: list of validation RMSEs over epochs
    train_spearmans: list of training Spearman rhos over epochs
    val_spearmans: list of validation Spearman rhos over epochs
    train_r2s: list of training R2s over epochs
    val_r2s: list of validation R2s over epochs
    val_preds: list of validation predictions
    val_targets: list of validation targets
    val_metrics: dictionary of validation metrics
'''
def train(train_df, seed, n_features_per_modality, model_tag, hyperparams=None):
    train_losses = []
    train_rmses = []
    val_rmses = []
    train_spearmans = []
    val_spearmans = []
    train_r2s = []
    val_r2s = []

    all_training_labels = []
    # Create DataLoader
    train_dataloader, val_dataloader = create_cross_validation_data_loaders(train_df, seed, model_tag, batch_size=hyperparams["batch_size"])
    for inputs, labels in train_dataloader:
        all_training_labels.append(labels)
    all_training_labels = torch.cat(all_training_labels)
    # Initialize model
    model = DCMFNet(
        NUM_MODALITIES, 
        hyperparams["num_layers"], 
        n_features_per_modality, 
        se_reduction=hyperparams["se_reduction"],
        dropout=hyperparams["dropout"],
        hidden_dim_min=hyperparams["hidden_dim_min"]
    ) 
    # using a custom loss function that has inverse frequency weighting and focal modulation to handle the imbalance in the regression labels and focus on harder samples
    criterion = ImbalancedRegressionLoss(
        all_training_labels,
        n_bins=hyperparams["n_bins"],
        focal_gamma=hyperparams["focal_gamma"],
        base_loss=hyperparams["base_loss"],
        huber_delta=hyperparams["huber_delta"]
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=hyperparams["learning_rate"], 
        weight_decay=hyperparams["weight_decay"]
    )  # Add weight decay for regularization

    num_epochs = hyperparams["num_epochs"]
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training on {len(train_dataloader.dataset)} samples...")
        # set the model to training mode
        model.train()  
        total_loss = 0.0
        i = 0


        for inputs, labels in train_dataloader:
            print(f"Batch {i+1}...")
            optimizer.zero_grad()
            # Forward pass
            #print(f"Forward pass for batch of size {inputs[0].size(0)}...")
            outputs = model(inputs) 
            #print(f"Shape of the outputs: {outputs.shape}")
            #print(f"Outputs: {outputs}")
            labels = labels.unsqueeze(1)  # Reshape to (batch_size, 1)
            print(f"Shape of the labels: {labels.shape}")
            #print(f"Labels: {labels}")
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Loss: {loss.item()}")
            predicted = outputs
            print(f"Shape of the Predicted: {predicted.shape}")
            #print(f"Predicted: {predicted}")
            i += 1

        avg_loss = total_loss / len(train_dataloader)

        # Evaluate on both sets
        print("  [Train]", end="")
        train_metrics, _, _ = evaluate(model, train_dataloader)
        print("  [Val]  ", end="")
        val_metrics, val_preds, val_targets = evaluate(model, val_dataloader)
        print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} | "
              f"Train rho: {train_metrics['spearman_rho']:.4f}, Val rho: {val_metrics['spearman_rho']:.4f} | "
              f"Train R2: {train_metrics['r2']:.4f}, Val R2: {val_metrics['r2']:.4f} |"
              f"Train RMSE: {train_metrics['rmse']:.4f}, Val RMSE: {val_metrics['rmse']:.4f} |"
              f"Train Pearson r: {train_metrics['pearson_r']:.4f}, Val Pearson r: {val_metrics['pearson_r']:.4f}")
 
        train_losses.append(avg_loss)
        train_rmses.append(train_metrics['rmse'])
        val_rmses.append(val_metrics['rmse'])
        train_spearmans.append(train_metrics['spearman_rho'])
        val_spearmans.append(val_metrics['spearman_rho'])
        train_r2s.append(train_metrics['r2'])
        val_r2s.append(val_metrics['r2'])
 
    return (model, train_losses, train_rmses, val_rmses, 
            train_spearmans, val_spearmans, train_r2s, val_r2s,
            val_preds, val_targets, val_metrics)



def evaluate_final_test(model, test_df, model_tag, seed, batch_size):
    X_test, Y_test = prepare_data(test_df, model_tag)
    test_dataloader = create_dataloader(X_test, Y_test, batch_size)
    test_metrics, test_preds, test_targets = evaluate(model, test_dataloader)
    print(f'\nFinal Test - RMSE: {test_metrics["rmse"]:.4f}, R2: {test_metrics["r2"]:.4f}, '
          f'Spearman rho: {test_metrics["spearman_rho"]:.4f}')
    plot_predicted_vs_actual(test_preds, test_targets, test_metrics, seed, model_tag, split_name="test")
    return test_metrics

if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape after dropping na: {df.shape}")
    modality_sizes = calculate_modality_sizes(df)
    #df = df.sample(frac=0.1, random_state=42)
    #print(f"Data shape: {df.shape}")
    seeds = [42] #[42, 43, 44, 45, 46]  # Example seeds for multiple runs
    for seed in seeds:
        torch.manual_seed(seed)
        train_df, test_df = random_split(df, test_size=0.25, random_state=seed)

        # Positive and negative symptom model training
        for model_tag in ["Pos", "Neg"]:
            print(f"\n{'='*60}")
            print(f"  Training {model_tag} symptom model")
            print(f"{'='*60}")
            start_time = time.time()
            hyperparams = hyperparameters_json[model_tag]

            (model, train_losses, train_rmses, val_rmses, 
                train_spearmans, val_spearmans, train_r2s, val_r2s,
                val_preds, val_targets, val_metrics) = train(train_df, seed, modality_sizes, model_tag, hyperparams)

            elapsed = (time.time() - start_time) / 60
            print(f"\nTime taken for {model_tag} model: {elapsed:.2f} minutes")

            plot_training_curves(train_losses, train_rmses, val_rmses,
                                    train_spearmans, val_spearmans,
                                    train_r2s, val_r2s, seed, model_tag)
            plot_predicted_vs_actual(val_preds, val_targets, val_metrics, 
                                        seed, model_tag, split_name="val")
            evaluate_final_test(model, test_df, model_tag, seed, batch_size=hyperparams["batch_size"])