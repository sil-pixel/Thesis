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


# Hyperparameters
learning_rate = 1.82e-4
batch_size = 6
num_epochs = 15

# Initialize model parameters
'''
6 modalities (input features)
5 iterative fusion layers (number of iterations for fusion) are found optimal in the paper,
 but can check for 1-7 and compare the performance of the model.
'''
num_modalities = 6  # change the number of modalities in the dataset if needed
num_layers = 5  # 5 is the default layers for now
fusion_iterations = np.arange(1, 8)  # Check for 1 to 7 iterations for fusion


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
def random_split(df, test_size=0.25, random_state=seed):
    unique_twin_ids = df['twin_id'].unique()
    train_ids, test_ids = train_test_split(unique_twin_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['twin_id'].isin(train_ids)]
    test_df = df[df['twin_id'].isin(test_ids)]
    return train_df, test_df


'''
Prepare the dataframe for training by splitting the dataframe into input and target modalities dataframes.
'''
def prepare_data(df):
    X = [df['SUD15'].values, df['PRS'].values, df.allmatches('SCZ15').values, df.allmatches('ATAC9').values, df.allmatches('ACE').values, df.allmatches('SES').values, df['Sex'].values, df['PCA'].values]
    Y = df['SCZ18'].values
    return X, Y


'''
Create a cross validation data loader with 20% of the data as the validation set and 80% of the data as the training set.
Input:
        df: pandas dataframe containing the training data
Output:
        train_dataloader: data loader containing the training data
        val_dataloader: data loader containing the validation data
        inputs: dataframe of shape (batch_size, num_features) where each column corresponds to a feature and each row corresponds to a sample.
'''
def create_cross_validation_data_loaders(df, seed):
    # attach the input and target modalities 
    train_df, val_df = random_split(df, test_size=0.2, random_state=seed)
    X_train, Y_train = prepare_data(train_df)
    X_val, Y_val = prepare_data(val_df)
    train_dataloader = create_dataloader(X_train, Y_train, batch_size)
    val_dataloader = create_dataloader(X_val, Y_val, batch_size)
    return train_dataloader, val_dataloader



'''
Create a data loader for the given input and target dataframes.
'''
def create_dataloader(X, Y, batch_size):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader



'''
Calculate the accuracy of the model on the given data.
Input:
    model: model to evaluate
    dataloader: data loader containing the data
Output:
    accuracy: accuracy of the model
'''
def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            _, predictions = torch.max(outputs, dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
    
    model.train()
    return (total_correct / total_samples) 


'''
Plot the training and validation accuracy and training loss curves and save the plots for each seed.
'''
def plot_training_curves(training_accuracies, validation_accuracies, train_losses, seed):
    epochs = range(1, len(training_accuracies) + 1)

    plt.figure(figsize=(12, 5))
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_accuracies, label='Training Accuracy')
    plt.plot(epochs, validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_seed_{seed}.png')
    plt.close()



'''
Train the model on the training data and evaluate the model on the test data.
Input:
    df: pandas dataframe containing the training data
Output:
    None
'''
def train(train_df, seed, plot_training=False):
    training_accuracies = []
    validation_accuracies = []
    train_losses = []
    # Create DataLoader
    train_dataloader, val_dataloader = create_cross_validation_data_loaders(train_df, seed)

    # Initialize model
    model = DCMFNet(num_modalities, num_layers)  
    # Define softmax Cross-entropy loss function and optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  
        total_loss = 0.0

        for train_inputs, train_targets in train_dataloader:
            train_inputs = train_inputs.float()  # Convert to float tensor
            train_targets = train_targets.float()  # Convert to float tensor

            optimizer.zero_grad()
            # Forward pass
            outputs = model(train_inputs) 
            loss = criterion(outputs, train_targets)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += train_targets.size(0)
            correct += (predicted == train_targets).sum().item()
        
        train_accuracy = correct / total

        avg_loss = total_loss / len(train_dataloader)
        # record the accuracy of the model
        val_accuracy = accuracy(model, val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {100 * train_accuracy:.4f}, Val Accuracy: {100 * val_accuracy:.4f}')
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        train_losses.append(avg_loss)
    
    if plot_training:
        plot_training_curves(training_accuracies, validation_accuracies, train_losses)
    
    return model


def evaluate_final_test(model, test_df):
    X_test, Y_test = prepare_data(test_df)
    test_dataloader = create_dataloader(X_test, Y_test, batch_size)
    test_accuracy = accuracy(model, test_dataloader)
    print(f'Final Test Accuracy: {100* test_accuracy:.4f}')


df = pd.read_csv('simulated_data.csv')
seeds = [42] #[42, 43, 44, 45, 46]  # Example seeds for multiple runs
for seed in seeds:
    torch.manual_seed(seed)
    train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
    model = train(train_df, seed, plot_training=False)
    #evaluate_final_test(model, test_df)