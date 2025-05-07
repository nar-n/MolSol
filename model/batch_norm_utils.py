"""
BatchNorm utilities for handling single samples during inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleBatchNorm1d(nn.Module):
    """
    BatchNorm1d that works with single samples during inference.
    This is a wrapper around BatchNorm1d that handles batch size 1 gracefully.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SingleBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.num_features = num_features
        
    def forward(self, x):
        if x.size(0) == 1 and self.training == False:
            # During inference with batch size 1, use instance norm behavior
            # Get the running mean and var from the BatchNorm layer
            running_mean = self.bn.running_mean
            running_var = self.bn.running_var
            
            # Apply normalization manually using the stored stats
            x = (x - running_mean.view(1, -1)) / torch.sqrt(running_var.view(1, -1) + self.bn.eps)
            
            # Apply affine transform if needed
            if self.bn.affine:
                x = x * self.bn.weight.view(1, -1) + self.bn.bias.view(1, -1)
            return x
        else:
            # Regular BatchNorm behavior for batch size > 1 or during training
            return self.bn(x)

def replace_bn_with_single_bn(module):
    """
    Recursively replace all BatchNorm1d layers with SingleBatchNorm1d
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, SingleBatchNorm1d(
                child.num_features, 
                child.eps, 
                child.momentum,
                child.affine,
                child.track_running_stats
            ))
            # Copy the parameters
            if child.affine:
                getattr(module, name).bn.weight.data = child.weight.data.clone()
                getattr(module, name).bn.bias.data = child.bias.data.clone()
            # Copy the running stats
            if child.track_running_stats:
                getattr(module, name).bn.running_mean.data = child.running_mean.data.clone()
                getattr(module, name).bn.running_var.data = child.running_var.data.clone()
        else:
            replace_bn_with_single_bn(child)
    
    return module
