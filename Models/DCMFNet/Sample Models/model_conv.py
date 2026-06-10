'''
Toy model for testing the iterative gated fusion model.
Author: Silpa Soni Nallacheruvu
Date: 11/02/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Steps:
1. Load toy data for training the model
2. Build basic building blocks of Iterative gated fusion model.
   2.1 Build the fusion module.
   2.2 Build the gated module.
3. Build the Deep Cross Modal Fusion Model.
'''
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass, field


'''
Performs the gated operation on the input tensor using the sigmoid activation function.
Input:
    F_curr: tensor of shape (batch_size, n_features_m) where n_features_m is the total number of features across the input modalitity 'M'
    G_prev: tensor of shape (batch_size, n_features_m)
Equation:
    z_t = σ(W_z (F_t + G_{t-1}) + b_z)
    f_t = F_t ⊙ z_t
    G_t = tanh(W_g([F_t; f_t]) + b_g)
    W_z and W_g represent the learnable parameters of convolution operations, 
    b_z and b_g represent the learnable bias terms, 
    and σ is the sigmoid activation function.
Output:
    G_curr: tensor of shape (batch_size, n_features_m)
'''
class GatedModule(nn.Module):
    def __init__(self, n_features_m):
        super().__init__()
        self.W_update = nn.Linear(n_features_m, n_features_m)  # Learnable parameter for the update operation
        self.W_gate = nn.Linear(n_features_m + n_features_m, n_features_m)  # Learnable parameter for the gated operation

    
    def forward(self, F_curr, G_prev):
        # Compute the gated operation on the input tensor
        z_curr = torch.sigmoid(self.W_update(F_curr + G_prev))
        f_curr = F_curr * z_curr
        G_curr = torch.tanh(self.W_gate(torch.cat((F_curr, f_curr), dim=1)))
        return G_curr

'''
Performs bilinear fusion of X and previous gated output using tanh activation function.
Input:
    X: tensor of shape (batch_size, n_features_x)
    G_prev: tensor of shape (batch_size, n_features_m)
    F_curr: tensor of shape (batch_size, n_features_m)
Equation:
    F_curr = tanh(sum (tanh(W1 @ X) * tanh(W2 @ G_prev))) 
    TODO: check what the sum operation is in the equation? 
    TODO: in the paper it was mentioned that the sum is integrating the multi-head output features along the channel dimension
Output:
    F_curr: tensor of shape (batch_size, n_features_m)
'''
class FusionModule(nn.Module):
    def __init__(self, n_features_x, n_features_m):
        super().__init__()
        self.W1 = nn.Linear(n_features_x, n_features_m)  # Learnable parameter for the first convolution operation in the fusion module
        self.W2 = nn.Linear(n_features_m, n_features_m)  # Learnable parameter for the second convolution operation in the fusion module

    
    def forward(self, X, G_prev):
        # Compute the fusion of X and G_prev
        F_curr = torch.tanh(torch.tanh(self.W1(X)) * torch.tanh(self.W2(G_prev)))
        return F_curr


'''
A single Gated Fusion Layer consisting of 1 fusion module and 1 gated module.
Input:
    X: tensor of shape (batch_size, n_features_x)
    G_prev: tensor of shape (batch_size, n_features_m) where n_features_m is the total number of features across the input modalitity 'M'
Output:
    F_curr: tensor of shape (batch_size, n_features_m)
    G_curr: tensor of shape (batch_size, n_features_m)
    
'''
class GatedFusionLayer(nn.Module):
    # TODO: choose the number of features for F_curr 
    def __init__(self, n_features_x, n_features_m):
        super().__init__()
        self.fusion_layer = FusionModule(n_features_x, n_features_m)
        self.gated_layer = GatedModule(n_features_m)

    
    def forward(self, X, G_prev):
        # Compute the fusion of X and G_prev
        F_curr = self.fusion_layer(X, G_prev)
        # Compute the gated operation on the input tensor
        G_curr = self.gated_layer(F_curr, G_prev)
        return F_curr, G_curr



'''
The Entire block of Iterative Gated Fusion Model consisting of 'L' Gated Fusion Layers.
Input:
    X: tensor of shape (batch_size, n_features_x)
    X_modalities: tensor of shape (batch_size, n_features_m) where n_features_m is the total number of features across the input modalitity 'M'
Equation:
    IGF = Conv([F_1; F_2; ...; F_L]) 
    where F_l is the output of the l-th Gated Fusion Layer 
    and Conv is a 1D convolution operation
    TODO: check the dimension along which the concatenation is performed in the equation.
Output:
    F_next: tensor of shape (batch_size, n_features_m * L)
'''
class IterativeGatedFusionModule(nn.Module):
    def __init__(self, L, n_features_x, n_features_m):
        super().__init__()
        self.L = L
        # using nn.ModuleList to store the Gated Fusion Layers
        self.gated_fusion_layers = nn.ModuleList([GatedFusionLayer(n_features_x, n_features_m) for _ in range(self.L)])
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  # Learnable parameter for the convolution operation in the Iterative Gated Fusion Module


    def forward(self, X, X_modality):
        G_prev = X_modality
        F_next = []
        for gated_fusion_layer in self.gated_fusion_layers:
            F_curr, G_prev = gated_fusion_layer(X, G_prev)
            F_next.append(F_curr)
        
        # Concatenate the outputs of all the Gated Fusion Layers
        F_next = torch.cat(F_next, dim=1) 
        # Add a channel dimension for convolution
        F_next = F_next.unsqueeze(1) 
        # Apply 1D convolution to the concatenated output
        return self.conv(F_next)



'''
The Deep Cross Modal Fusion Model consisting of 'M' Iterative Gated Fusion Modules and a final fully connected layer for prediction.
Variables:
    M: number of Iterative Gated Fusion Modules
    L: number of Gated Fusion Layers in each Iterative Gated Fusion Module
    n_features_x: number of features in the input modality 'X'
    n_features_per_modality: list of number of features in each of the 'M' modalities, can be changed based on the dataset used.
    conv_fused: learnable parameter for the convolution operation on the fused output of all the Iterative Gated Fusion Modules
    conv_independent: learnable parameter for the convolution operation on the independent modalities
    conv_final: learnable parameter for the convolution operation on the concatenated output of the fused output and independent modalities
    fc: fully connected layer for prediction, in_features can be changed based on the output of the convolution operation on the concatenated output of the fused output and independent modalities
'''
class DeepCrossModalFusionModel(nn.Module):
    def __init__(self, M, L, n_features_x, n_features_per_modality):
        super().__init__()
        self.M = M
        self.igf_modules = nn.ModuleList([
            IterativeGatedFusionModule(L, n_features_x, n_features_per_modality[m])
             for m in range(self.M)])
        self.conv_fused = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  
        self.conv_independent = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  
        self.conv_final = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  
        total_final_features =  L * sum(n_features_per_modality[:M]) + n_features_x + sum(n_features_per_modality) # Total features after concatenating the fused output and independent modalities
        #print(f"Total features after concatenating the fused output and independent modalities: {total_final_features}")
        self.fc = nn.Linear(in_features=total_final_features, out_features=1)  # Learnable parameter for the fully connected layer for prediction

    def forward(self, inputs):
        #print("Training the Deep Cross Modal Fusion Model...")
        X = inputs[0]
        modalities = inputs[1:self.M+1]  # Get the 'M' modalities from the input list
        X_ind = inputs[self.M+1]  # Get the independent modality 'X7' from the input list

        # Pass the input modality 'X' and each of the 'M' modalities through the 'M' Iterative Gated Fusion Modules
        F_out = []
        for m, igf in enumerate(self.igf_modules):
            #print(f"Processing modality {m+1} through Iterative Gated Fusion Module {m+1}...")
            #print(f"Input shape for modality {m+1}: {modalities[m].shape}")
            F_out.append(igf(X, modalities[m]))
        
        #print(f"Shape of the output from each Iterative Gated Fusion Module: {[f.shape for f in F_out]}")
        F_out = torch.cat(F_out, dim=-1)  # Concatenate the outputs of all the Iterative Gated Fusion Modules
        #print(f"Shape of the concatenated output from all Iterative Gated Fusion Modules: {F_out.shape}")
        F_out = self.conv_fused(F_out)  # Apply convolution to the fused output of all the Iterative Gated Fusion Modules
        #print(f"Shape of the fused output after convolution: {F_out.shape}")

        # Pass independent modalities through the fully connected layer for prediction
        #print(f"Shape of all independent modalities before convolution: {X_ind.shape}")
        #print(f"Shape of the input modality 'X' before convolution: {X.shape}")
        #print(f"Shape of the modalities before convolution: {[modality.shape for modality in modalities]}")
        X_independent = torch.cat([X] + modalities + [X_ind], dim=-1)  # Concatenate all the independent modalities
        #print(f"Shape of the concatenated independent modalities before convolution: {X_independent.shape}")
        X_independent = X_independent.unsqueeze(1)
        #print(f"Shape of the concatenated independent modalities after adding channel dimension: {X_independent.shape}")
        X_independent = self.conv_independent(X_independent)
        #print(f"Shape of the independent modalities after convolution: {X_independent.shape}")


        # Concatenate the outputs of the independent modalities and the convolutional fusion
        F_final = torch.cat((F_out, X_independent), dim=-1)
        #print(f"Shape of the concatenated output of the fused output and independent modalities before convolution: {F_final.shape}")
        F_final = self.conv_final(F_final)
        #print(f"Shape of the concatenated output of the fused output and independent modalities after convolution: {F_final.shape}")

        # Pass through the fully connected layer for prediction
        F_final = F_final.squeeze(1)  # Remove the channel dimension before passing to the fully connected layer
        #print(f"F_final: {F_final}")
        #print(f"Shape of the output after removing channel dimension: {F_final.shape}")
        output = self.fc(F_final)  # Remove the channel dimension before passing to the fully connected layer
        #output = torch.sigmoid(output)  # Apply sigmoid activation to get the final output in the range [0, 1]
        return output


    
