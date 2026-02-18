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



'''
Performs the gated operation on the input tensor using the sigmoid activation function.
Input:
    F_curr: tensor of shape (batch_size, n_features_f)
    G_prev: tensor of shape (batch_size, n_features_g)
Equation:
    z_t = σ(W_z (F_t + G_{t-1}) + b_z)
    f_t = F_t ⊙ z_t
    G_t = tanh(W_g([F_t; f_t]) + b_g)
    W_z and W_g represent the learnable parameters of convolution operations, 
    b_z and b_g represent the learnable bias terms, 
    and σ is the sigmoid activation function.
Output:
    G_curr: tensor of shape (batch_size, n_features_g)
'''
@dataclass
class GatedModule:
    def __init__(self, F_curr, G_prev, W_gate, b_gate, W_update, b_update):
        self.F_curr = F_curr
        self.G_prev = G_prev
        self.W_gate = W_gate
        self.b_gate = b_gate
        self.W_update = W_update
        self.b_update = b_update
    
    def forward(self):
        # Compute the gated operation on the input tensor
        z_curr = torch.sigmoid(self.W_update @ (self.F_curr + self.G_prev) + self.b_update)
        f_curr = self.F_curr * z_curr
        G_curr = torch.tanh(self.W_gate @ torch.cat((self.F_curr, f_curr), dim=1) + self.b_gate)
        return G_curr

'''
Performs bilinear fusion of X and previous gated output using tanh activation function.
Input:
    X: tensor of shape (batch_size, n_features_x)
    G_prev: tensor of shape (batch_size, n_features_g)
    F_curr: tensor of shape (batch_size, n_features_f)
Equation:
    F_curr = tanh(sum (tanh(W1 @ X) * tanh(W2 @ G_prev)) )
    TODO: check what the sum operation is in the equation? 
Output:
    F_curr: tensor of shape (batch_size, n_features_f)
'''
@dataclass
class FusionModule:
    def __init__(self, X, G_prev, W1, W2):
        self.X = X
        self.G_prev = G_prev
        self.W1 = W1
        self.W2 = W2
    
    def forward(self):
        # Compute the fusion of X and G_prev
        F_curr = torch.tanh(torch.tanh(self.W1 @ self.X) * torch.tanh(self.W2 @ self.G_prev))
        return F_curr




'''
A single Gated Fusion Layer consisting of 1 fusion module and 1 gated module.
'''
@dataclass
class GatedFusionLayer:
    def __init__(self, X, G_prev):
        self.X = X
        self.G_prev = G_prev
        self.W1 = # TODO: Initialize W1
        self.W2 = # TODO: Initialize W2
        self.W_gate = # TODO: Initialize W_gate
        self.b_gate = # TODO: Initialize b_gate to zeroes
        self.W_update = # TODO: Initialize W_update
        self.b_update = # TODO: Initialize b_update to zeroes

    
    def forward(self):
        # Compute the fusion of X and G_prev
        Fusion = FusionModule(self.X, self.G_prev, self.W1, self.W2)
        F_curr = Fusion.forward()
        # Compute the gated operation on the input tensor
        Gate = new GatedModule(F_curr, self.G_prev, self.W_gate, self.b_gate, self.W_update, self.b_update)
        G_curr = Gate.forward()
        return F_curr, G_curr



'''
The Entire block of Iterative Gated Fusion Model consisting of 'L' Gated Fusion Layers.
Input:
    X: tensor of shape (batch_size, n_features_x)
    Y: tensor of shape (batch_size, n_features_y)
Equation:
    IGF = Conv([F_1; F_2; ...; F_L]) 
    where F_l is the output of the l-th Gated Fusion Layer 
    and Conv is a 1D convolution operation
    TODO: check the dimension along which the concatenation is performed in the equation.
Output:
    F_next: tensor of shape (batch_size, n_features_f * L)
'''
@dataclass
class IterativeGatedFusionModule:
    def __init__(self, GatedFusionLayer, L):
        self.GatedFusionLayer = GatedFusionLayer
        self.L = L

    def forward(self, X, Y):
        G_prev = Y
        F_next = []
        for l in range(self.L):
            F_curr, G_prev = self.GatedFusionLayer(X, G_prev)
            F_next.append(F_curr)
        
        # Concatenate the outputs of all the Gated Fusion Layers
        F_next = torch.cat(F_next, dim=1) 
        # Add a channel dimension for convolution
        F_next = F_next.unsqueeze(1) 
        # Apply 1D convolution to the concatenated output
        F_out = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)(F_next) 
        return F_out



'''
The Deep Cross Modal Fusion Model consisting of 'M' Iterative Gated Fusion Modules and a final fully connected layer for prediction.
'''
@dataclass
class DeepCrossModalFusionModel:
    def __init__(self, M, L):
        self.M = M
        self.L = L
        self.GatedFusionLayer = GatedFusionLayer()
        self.IterativeGatedFusionModule = IterativeGatedFusionModule(self.GatedFusionLayer, self.L)
        self.fc = nn.Linear(in_features=self.L, out_features=1)
    
    def train(self, X, Y1, Y2, Y3, Y4, Y5, Y6):
        Y = [Y1, Y2, Y3, Y4, Y5, Y6]  # List of all the input modalities
        F_out = []
        for m in range(self.M):
            F_next = self.IterativeGatedFusionModule(X, Y[m])
            F_out.append(F_next)

        # Convolutional fusion of the outputs from all the Iterative Gated Fusion Modules
        F_out = torch.cat(F_out, dim=1)  # Concatenate along the channel dimension
        F_out = F_out.unsqueeze(1)
        F_out = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)(F_out)

        # Pass through the fully connected layer for prediction
        output = self.fc(F_out)
        return output


    
