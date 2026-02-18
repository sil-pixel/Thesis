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

# Load toy data
df = pd.read_csv('simulated_data.csv')


# train the model
def train():
    # Create DataLoader
    dataset = list(zip(df['input'], df['target']))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = DCMFNet(num_modalities, num_iteranum_layers)  
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # TODO: check Sigmoid Cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
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


