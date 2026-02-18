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

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)


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


# split the data into test and train sets
def random_split(df, test_size=0.25, random_state=seed):
    unique_twin_ids = df['twin_id'].unique()
    train_ids, test_ids = train_test_split(unique_twin_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['twin_id'].isin(train_ids)]
    test_df = df[df['twin_id'].isin(test_ids)]
    return train_df, test_df

# train the model
def train(df):
    # Create DataLoader
    dataloader = DataLoader(df, batch_size=batch_size, shuffle=True, 
    generator=torch.Generator().manual_seed(seed))  # Set random seed for reproducibility
    X = df.allmatches('substance_use').values 
    Y1 = df['PRS'].values
    Y2 = df.allmatches('substance_use').values 
    Y3 = df.allmatches('ATAC').values 
    Y4 = df.allmatches('ACE').values 
    Y5 = df.allmatches('SES').values 
    Y6 = df['Sex'].values

    # Initialize model
    model = DCMFNet(num_modalities, num_layers)  
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # TODO: check Sigmoid Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train(X, Y1, Y2, Y3, Y4, Y5, Y6)  
        total_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.float()  # Convert to float tensor
            targets = targets.float()  # Convert to float tensor

            # Forward pass
            outputs = model(inputs) # TODO: check 
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


def main():
    df = pd.read_csv('simulated_data.csv')
    train_df, test_df = random_split(df)
    train(train_df)