'''
Toy model for testing the iterative gated fusion model with Soft Attention and Squeeze-and-Excitation (SE) Attention mechanisms.
Author: Silpa Soni Nallacheruvu
Date: 09/04/2026
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
Learnable soft attention over features of a vector.

Given input x of shape (batch, D):
    1. score_i = W2 @ tanh(W1 @ x_i + b1) + b2   for each feature i
    2. alpha = softmax(scores)                      attention weights
    3. output = x * alpha                           weighted features

This preserves the full dimensionality (unlike pooling) while letting 
the model learn to emphasize or suppress individual features.

Parameters:
    n_features: the dimension D of the input vector
    hidden_dim: internal dimension of the scoring network (default: D//2, min 8)
    dropout: dropout on attention weights (default: 0.3)
'''

class SoftAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=None, dropout=0.3):
        super().__init__()
        self.n_features = n_features
        if hidden_dim is None:
            hidden_dim = max(n_features // 2, 8)
        
        # Scoring network: maps each feature through a shared 2-layer network
        self.score_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Context-aware gating: use global context to modulate attention
        # This lets the model consider ALL features when deciding how to 
        # weight each one, unlike Conv1d which was purely element-wise
        self.context_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Args:
            x: (batch, D)
        Returns:
            out: (batch, D) — attention-weighted features
        '''
        # Context-aware modulation: (batch, D)
        context = self.context_net(x)
        
        # Compute per-feature attention scores
        # Reshape to (batch, D, 1) so each feature is scored independently
        x_expanded = context.unsqueeze(-1)  # (batch, D, 1)
        scores = self.score_net(x_expanded).squeeze(-1)  # (batch, D)
        
        # Normalize to attention weights
        alpha = F.softmax(scores, dim=-1)  # (batch, D)
        alpha = self.dropout(alpha)
        
        # Apply attention: element-wise weighting
        out = x * alpha * self.n_features  # scale by D to preserve magnitude
        # (without scaling, softmax would shrink values since weights sum to 1)
        return out


'''
The Entire block of Iterative Gated Fusion Model consisting of 'L' Gated Fusion Layers.
Input:
    X: tensor of shape (batch_size, n_features_x)
    X_modalities: tensor of shape (batch_size, n_features_m) where n_features_m is the total number of features across the input modalitity 'M'
Equation:
    IGF = SoftAttention([F_1; F_2; ...; F_L]) 
    where F_l is the output of the l-th Gated Fusion Layer 
    Changed: Conv1d -> SoftAttention over the concatenated L fusion outputs.
    SoftAttention allows the model to learn to emphasize or suppress individual features across the concatenated output of all the Gated Fusion Layers, while preserving the full dimensionality of the output.
Output:
    F_next: tensor of shape (batch_size, n_features_m * L)
'''
class IterativeGatedFusionModule(nn.Module):
    def __init__(self, L, n_features_x, n_features_m, dropout=0.3):
        super().__init__()
        self.L = L
        # using nn.ModuleList to store the Gated Fusion Layers
        self.gated_fusion_layers = nn.ModuleList([GatedFusionLayer(n_features_x, n_features_m) for _ in range(self.L)])
        self.attention = SoftAttention(n_features_m * self.L, dropout=dropout)  # Learnable parameter for the attention operation on the concatenated output of all the Gated Fusion Layers


    def forward(self, X, X_modality):
        G_prev = X_modality
        F_next = []
        for gated_fusion_layer in self.gated_fusion_layers:
            F_curr, G_prev = gated_fusion_layer(X, G_prev)
            F_next.append(F_curr)
        
        # Concatenate the outputs of all the Gated Fusion Layers
        F_next = torch.cat(F_next, dim=1) 
        # Apply attention to the concatenated output of all the Gated Fusion Layers
        F_next = self.attention(F_next)
        #print(f"Shape of the output from the Iterative Gated Fusion Module after attention: {F_next.shape}")
        return F_next



'''
Squeeze-and-Excitation (SE) Attention module.

Adapts the channel-wise SE block (Hu et al., 2018) to work on 1D feature 
vectors in tabular/multi-modal settings.

How it works:
  1. Squeeze:  Global average pooling collapses the input to a single scalar 
               per feature — captures "how active is this feature on average."
               For a 1D vector (batch, D), this is just the input itself, so 
               squeeze is implicit.
  2. Excitation: A bottleneck MLP (D -> D//r -> D) learns channel-wise 
                 importance weights, activated by sigmoid to produce gates 
                 in [0, 1].
  3. Scale:    Element-wise multiply input by the gates.

Why this helps at the final layer of DCMFNet:
  - The final concatenated vector has hundreds of features from very different 
    sources (fused outputs + raw modalities). SE lets the model learn to 
    recalibrate which of those features matter for prediction.
  - Unlike SoftAttention (softmax weights sum to 1, competitive), SE uses 
    sigmoid (each gate is independent), so it can boost multiple features 
    simultaneously or suppress them independently.

Interface:
  Input:  (batch, D)
  Output: (batch, D)

Parameters:
    n_features: dimension D of the input vector
    reduction: bottleneck reduction ratio (default: 2). 
                Hidden dim = max(D // reduction, 8).
    dropout: dropout applied inside the excitation path (default: 0.3)
'''

class SEAttention(nn.Module):

    def __init__(self, n_features, reduction=2, dropout=0.3):
        super().__init__()
        hidden_dim = max(n_features // reduction, 8)

        self.excitation = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Args:
            x: (batch, D)
        Returns:
            out: (batch, D) — recalibrated features
        '''
        # Squeeze is implicit for 1D vectors (no spatial dims to pool over)
        # Excitation: learn per-feature gates
        gates = self.excitation(x)  # (batch, D), values in [0, 1]
        # Scale: element-wise recalibration
        return x * gates

'''
The Deep Cross Modal Fusion Model consisting of 'M' Iterative Gated Fusion Modules and a final fully connected layer for prediction.
Variables:
    M: number of Iterative Gated Fusion Modules
    L: number of Gated Fusion Layers in each Iterative Gated Fusion Module
    n_features_x: number of features in the input modality 'X'
    n_features_per_modality: list of number of features in each of the 'M' modalities, can be changed based on the dataset used.
    dropout: dropout rate for the attention layers (default: 0.3)
    attn_fused: learnable parameter for the attention operation on the fused output of all the Iterative Gated Fusion Modules
    attn_independent: learnable parameter for the attention operation on the independent modalities
    attn_final: learnable parameter for the attention operation on the concatenated output of the fused output and independent modalities
    fc: fully connected layer for prediction, in_features is the total number of features after concatenating the fused output and independent modalities, out_features is 1 for continuous output for prediction.
'''
class DeepCrossModalFusionModel(nn.Module):
    def __init__(self, M, L, n_features_x, n_features_per_modality, dropout=0.3):
        super().__init__()
        self.M = M
        self.igf_modules = nn.ModuleList([
            IterativeGatedFusionModule(L, n_features_x, n_features_per_modality[m])
             for m in range(self.M)])
        
        # Dimensions for each attention layer
        fused_dim = L * sum(n_features_per_modality[:M])
        independent_dim = n_features_x + sum(n_features_per_modality)
        total_final_features = fused_dim + independent_dim
        self.attn_fused =  SEAttention(fused_dim, dropout=dropout)  
        self.attn_independent = SEAttention(independent_dim, dropout=dropout)
        self.attn_final = SEAttention(total_final_features, dropout=dropout) 
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
        F_out = self.attn_fused(F_out)  # Apply attention to the concatenated output of all the Iterative Gated Fusion Modules
        #print(f"Shape of the fused output after attention: {F_out.shape}")

        # Pass independent modalities through the fully connected layer for prediction
        #print(f"Shape of all independent modalities before attention: {X_ind.shape}")
        #print(f"Shape of the input modality 'X' before attention: {X.shape}")
        #print(f"Shape of the modalities before attention: {[modality.shape for modality in modalities]}")
        X_independent = torch.cat([X] + modalities + [X_ind], dim=-1)  # Concatenate all the independent modalities
        #print(f"Shape of the concatenated independent modalities before attention: {X_independent.shape}")
        X_independent = self.attn_independent(X_independent)  # Apply attention to the concatenated independent modalities
        #print(f"Shape of the independent modalities after attention: {X_independent.shape}")


        # Concatenate the outputs of the independent modalities and the convolutional fusion
        F_final = torch.cat((F_out, X_independent), dim=-1)
        #print(f"Shape of the concatenated output of the fused output and independent modalities before attention: {F_final.shape}")
        F_final = self.attn_final(F_final)  # Apply attention to the concatenated output of the fused output and independent modalities
       #print(f"Shape of the concatenated output of the fused output and independent modalities after attention: {F_final.shape}")

        # Pass through the fully connected layer for prediction
        output = self.fc(F_final) 
        #print(f"Shape of the output from the fully connected layer after activation: {output.shape}")
        #print(f"Output from the fully connected layer after activation: {output}")
        #output = torch.sigmoid(output)  # Apply sigmoid activation to get the final output in the range [0, 1]
        return output


    
