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
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Hyperparameters
learning_rate = 1.82e-4
batch_size = 8
num_epochs = 20

# Add regularization techniques such as dropout and weight decay to prevent overfitting, especially given the small batch size and number of epochs.
weight_decay = 1e-3  # Example weight decay for L2 regularization

# Initialize model parameters
'''
7 modalities (input features)
5 iterative fusion layers (number of iterations for fusion) are found optimal in the paper,
 but can check for 1-7 and compare the performance of the model.
'''
num_modalities = 9 
num_layers = 5  # 5 is the default layers for now
fusion_iterations = np.arange(1, 8)  # Check for 1 to 7 iterations for fusion

# Known number of output features in the dataset
y_features = {
    "Pos" : 24,
    "Neg": 11
}

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
        df.filter(regex="^PC").to_numpy()
    ]
    if model_tag == "Pos":
        Y = df["SCZ18_Pos"].to_numpy()
    elif model_tag == "Neg":
        Y = df["SCZ18_Neg"].to_numpy()

    
    return X, Y

'''
Calculate the modality sizes for the catss dataset
'''
def calculate_modality_sizes(df):
    prefixes = ["SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX", "PC"]
    counts = [
        sum(col.startswith(prefix) for col in df.columns)
        for prefix in prefixes
    ]
    print(f"modality sizes for {prefixes} : {counts}")
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
def create_cross_validation_data_loaders(df, seed, model_tag):
    # attach the input and target modalities 
    train_df, val_df = random_split(df, test_size=0.2, random_state=seed)
    X_train, Y_train = prepare_data(train_df, model_tag)
    X_val, Y_val = prepare_data(val_df, model_tag)
    train_dataloader = create_dataloader(X_train, Y_train, batch_size)
    val_dataloader = create_dataloader(X_val, Y_val, batch_size)
    return train_dataloader, val_dataloader

'''
Calculate the accuracy of the model on the given data.
Input:
    model: model to evaluate
    dataloader: data loader containing the data
Output:
    accuracy: accuracy of the model
'''
def accuracy(model, dataloader, model_tag):
    model.eval()
    all_predictions = []
    all_targets = []
    n_features_y = y_features.get(model_tag)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets = targets.unsqueeze(1)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    
    # Check for constant predictions before Spearman
    if np.std(all_predictions) < 1e-6:
        print("WARNING: predictions are constant, Spearman undefined")
        rho, p = float('nan'), float('nan')
    else:
        rho, p = spearmanr(all_targets, all_predictions)
    
    correct = (np.abs(all_predictions - all_targets) < 0.05).sum()
    accuracy_score = correct / len(all_targets)
    
    print(f"  Prediction std: {np.std(all_predictions):.6f}")
    print(f"  Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
    print(f"  MAE: {mae:.4f}, MSE: {mse:.4f}")
    print(f"  Spearman rho: {rho:.4f}, p-value: {p:.4f}")
    
    model.train()
    return accuracy_score, mae


'''
Plot the training and validation accuracy and training loss curves and save the plots for each seed.
'''
def plot_training_curves(training_accuracies, validation_accuracies, train_losses, training_maes, validation_maes, seed, model_tag):
    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(15, 5))
    
    # Plot training and validation accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot training loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot training and validation MAE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, training_maes, label='Training MAE')
    plt.plot(epochs, validation_maes, label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_tag}_training_curves_seed_{seed}.png')
    plt.close()
    print(f"Training curves plotted and saved to '{model_tag}_training_curves_seed_{seed}.png'")


'''
Train the model on the training data and evaluate the model on the test data.
Input:
    df: pandas dataframe containing the training data
    n_features_per_modality: List of number of features in each modality
    model_tag: tag to distinguihs between positive or negative schizophrenic symptom model
Output:
    None
'''
def train(train_df, seed, n_features_per_modality, model_tag):
    training_accuracies = []
    validation_accuracies = []
    train_losses = []
    training_maes = []
    validation_maes = []
    # Create DataLoader
    train_dataloader, val_dataloader = create_cross_validation_data_loaders(train_df, seed, model_tag)
    n_features_y = y_features.get(model_tag)
    # Initialize model
    model = DCMFNet(num_modalities, num_layers, n_features_per_modality) 
    # define MSE loss for a regression task and Adam optimizer with weight decay for regularization
    criterion = nn.MSELoss(reduction='none')  # Use mean squared error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Add weight decay for regularization

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training on {len(train_dataloader.dataset)} samples...")
        # set the model to training mode
        model.train()  
        total_loss = 0.0
        total = 0
        correct = 0
        i = 0
        total_mae = 0.0


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
            #sample_weights = 1.0 + 3.0 * labels.squeeze()  # Example: higher weight for higher labels, adjust as needed
            loss = (criterion(outputs, labels).squeeze() * sample_weights).mean()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Loss: {loss.item()}")
            predicted = outputs
            print(f"Shape of the Predicted: {predicted.shape}")
            #print(f"Predicted: {predicted}")
            total += labels.size(0)  # Total number of samples (batch_size)
            correct += (torch.abs(predicted - labels) < 0.05).sum().item()
            print(f"Correct: {correct}, Total: {total}")
            mae = mean_absolute_error(labels.cpu().numpy(), predicted.detach().cpu().numpy())
            print(f"Mean Absolute Error: {mae}")
            total_mae += mae
            i += 1
        train_accuracy = correct / total

        avg_loss = total_loss / len(train_dataloader)
        total_mae = total_mae / len(train_dataloader)
        # record the accuracy of the model
        val_accuracy, val_mae = accuracy(model, val_dataloader, model_tag)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {100 * train_accuracy:.4f}, Val Accuracy: {100 * val_accuracy:.4f}', f'Training MAE: {total_mae:.4f}, Validation MAE: {val_mae:.4f}')
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        train_losses.append(avg_loss)
        training_maes.append(total_mae)
        validation_maes.append(val_mae)
    return model, training_accuracies, validation_accuracies, train_losses, training_maes, validation_maes



def evaluate_final_test(model, test_df, model_tag):
    X_test, Y_test = prepare_data(test_df, model_tag)
    test_dataloader = create_dataloader(X_test, Y_test, batch_size)
    test_accuracy, test_mae = accuracy(model, test_dataloader, model_tag)
    print(f'Final Test Accuracy: {100 * test_accuracy:.4f}, Final Test MAE: {test_mae:.4f}')



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

        # Positive symptom model training
        model_tag = "Pos"
        start_time = time.time()
        pos_model, pos_training_accuracies, pos_validation_accuracies, pos_train_losses, pos_training_maes, pos_validation_maes = train(train_df, seed, modality_sizes, model_tag)
        end_time = time.time()
        print(f"Time taken for the positive SCZ model: {(end_time - start_time)/60:.2f} minutes")
        plot_training_curves(pos_training_accuracies, pos_validation_accuracies, pos_train_losses, pos_training_maes, pos_validation_maes, seed, model_tag)
        #evaluate_final_test(pos_model, test_df, y_pos_features)

        # Negative symptom model training
        model_tag = "Neg"
        start_time = time.time()
        neg_model, neg_training_accuracies, neg_validation_accuracies, neg_train_losses, neg_training_maes, neg_validation_maes = train(train_df, seed, modality_sizes, model_tag)
        end_time = time.time()
        print(f"Time taken for the negative SCZ model: {(end_time - start_time)/60:.2f} minutes")
        plot_training_curves(neg_training_accuracies, neg_validation_accuracies, neg_train_losses, neg_training_maes, neg_validation_maes, seed, model_tag)
        #evaluate_final_test(neg_model, test_df, y_neg_features)