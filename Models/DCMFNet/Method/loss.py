'''
Custom loss functions for imbalanced continuous regression targets.

Problem: With right-skewed target distributions (most values near 0, 
rare high values indicating SCZ symptoms), standard MSE lets the model 
"get away with" predicting the dense region mean (~0.1-0.2) for everything.

Solution: Two complementary strategies combined into one loss.

1. Inverse-frequency bin weighting: 
   Weights each sample inversely proportional to how many samples share 
   its target range. Rare high-value SCZ cases get much higher weight.

2. Focal regression loss: 
   Downweights easy (small error) samples, upweights hard (large error) 
   samples. Since the tail cases are harder to predict, they naturally 
   get more gradient signal.

Usage:
    # Before training
    loss_fn = ImbalancedRegressionLoss(
        all_train_labels,   # all normalized training labels as 1D tensor
        n_bins=10,
        focal_gamma=2.0,
        base_loss='mse'     # or 'huber'
    )
    
    # Inside training loop
    loss = loss_fn(predictions, labels)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalancedRegressionLoss(nn.Module):
    '''
    Combined inverse-frequency + focal regression loss.
    
    Args:
        train_labels: 1D tensor of all normalized training labels (0 to 1),
                      used to compute bin frequencies.
        n_bins: number of bins for frequency estimation (default: 10)
        focal_gamma: focal exponent. 0 = no focal effect, 
                     higher = more focus on hard samples (default: 2.0)
        max_weight: cap on bin weights to prevent instability (default: 20.0)
        base_loss: 'mse' or 'huber' (default: 'mse')
        huber_delta: delta for huber loss if used (default: 0.1)
    '''
    def __init__(self, train_labels, n_bins=10, focal_gamma=2.0, 
                 max_weight=20.0, base_loss='mse', huber_delta=0.1):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.base_loss_type = base_loss
        self.huber_delta = huber_delta
        
        # Compute bin edges and inverse-frequency weights from training data
        self.n_bins = n_bins
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer('bin_edges', bin_edges)
        
        # Count samples per bin
        bin_indices = torch.bucketize(train_labels.detach(), bin_edges) - 1
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        bin_counts = torch.bincount(bin_indices, minlength=n_bins).float()
        
        # Inverse frequency: rare bins get high weight
        inv_freq = 1.0 / (bin_counts + 1.0)  # +1 smoothing to avoid div by zero
        # Normalize so mean weight = 1 (preserves loss scale)
        inv_freq = inv_freq / inv_freq.mean()
        # Cap extreme weights for stability
        inv_freq = inv_freq.clamp(max=max_weight)
        
        self.register_buffer('bin_weights', inv_freq)
        
        # Print weight distribution for sanity check
        print(f"\n--- ImbalancedRegressionLoss initialized ---")
        print(f"  Bins: {n_bins}, Focal gamma: {focal_gamma}, Base loss: {base_loss}")
        for i in range(n_bins):
            edge_lo = bin_edges[i].item()
            edge_hi = bin_edges[i + 1].item()
            count = bin_counts[i].item()
            weight = inv_freq[i].item()
            print(f"  Bin [{edge_lo:.2f}, {edge_hi:.2f}): "
                  f"count={int(count):>5}, weight={weight:.3f}")
        print(f"-------------------------------------------\n")
    
    def _get_sample_weights(self, labels):
        '''Look up the bin weight for each sample based on its label value.'''
        bin_indices = torch.bucketize(labels.detach(), self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.n_bins - 1)
        return self.bin_weights[bin_indices]
    
    def forward(self, predictions, targets):
        '''
        Args:
            predictions: (batch,) or (batch, 1)
            targets: (batch,) or (batch, 1)
        Returns:
            scalar loss
        '''
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        
        # Per-sample base loss
        if self.base_loss_type == 'huber':
            per_sample_loss = F.huber_loss(
                predictions, targets, 
                reduction='none', delta=self.huber_delta
            )
        else:
            per_sample_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Focal modulation: downweight easy samples, upweight hard ones
        # normalized_error is in [0, 1] roughly (since targets are in [0, 1])
        with torch.no_grad():
            abs_error = torch.abs(predictions - targets)
            # Focal weight: (error)^gamma — larger errors get more weight
            focal_weight = (abs_error + 1e-6) ** self.focal_gamma
            # Normalize focal weights so they don't change overall loss scale
            focal_weight = focal_weight / (focal_weight.mean() + 1e-6)
        
        # Inverse-frequency bin weights
        bin_weight = self._get_sample_weights(targets)
        
        # Combine: both focal and bin weights
        combined_weight = focal_weight * bin_weight
        # Renormalize so mean weight ≈ 1
        combined_weight = combined_weight / (combined_weight.mean() + 1e-6)
        
        # Weighted loss
        loss = (per_sample_loss * combined_weight).mean()
        return loss


class InverseFrequencyMSELoss(nn.Module):
    '''
    Simpler variant: just inverse-frequency weighting without focal component.
    Use this if focal loss makes training unstable.
    
    Args:
        train_labels: 1D tensor of all normalized training labels
        n_bins: number of bins (default: 10)
        max_weight: cap on bin weights (default: 20.0)
    '''
    def __init__(self, train_labels, n_bins=10, max_weight=20.0):
        super().__init__()
        self.n_bins = n_bins
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer('bin_edges', bin_edges)
        
        bin_indices = torch.bucketize(train_labels.detach(), bin_edges) - 1
        bin_indices = bin_indices.clamp(0, n_bins - 1)
        bin_counts = torch.bincount(bin_indices, minlength=n_bins).float()
        
        inv_freq = 1.0 / (bin_counts + 1.0)
        inv_freq = inv_freq / inv_freq.mean()
        inv_freq = inv_freq.clamp(max=max_weight)
        self.register_buffer('bin_weights', inv_freq)
        
        print(f"\n--- InverseFrequencyMSELoss initialized ---")
        for i in range(n_bins):
            print(f"  Bin [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): "
                  f"count={int(bin_counts[i]):>5}, weight={inv_freq[i]:.3f}")
        print()
    
    def forward(self, predictions, targets):
        predictions = predictions.squeeze()
        targets = targets.squeeze()
        per_sample_loss = F.mse_loss(predictions, targets, reduction='none')
        bin_indices = torch.bucketize(targets.detach(), self.bin_edges) - 1
        bin_indices = bin_indices.clamp(0, self.n_bins - 1)
        weights = self.bin_weights[bin_indices]
        return (per_sample_loss * weights).mean()