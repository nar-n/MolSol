import torch
import torch.nn as nn
from torch.nn import functional as F
from model.gnn_encoder import GNNEncoder
from model.batch_norm_utils import SingleBatchNorm1d

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, n_tasks=1, dropout=0.3):
        super(PropertyPredictor, self).__init__()
        
        self.dropout = dropout
        
        # Enhanced prediction MLP with our custom batch normalization for single samples
        self.fc1 = nn.Linear(latent_dim, hidden_dim*2)
        self.bn1 = SingleBatchNorm1d(hidden_dim*2)
        
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn2 = SingleBatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn3 = SingleBatchNorm1d(hidden_dim//2)
        
        # Output layer
        self.output = nn.Linear(hidden_dim//2, n_tasks)
        
        # Residual connection - fixed dimension to match
        self.res_connection = nn.Linear(latent_dim, hidden_dim//2)
    
    def forward(self, z):
        # Store original input for residual connection
        identity = self.res_connection(z)
        
        # First block
        h = self.fc1(z)
        h = self.bn1(h)
        h = F.leaky_relu(h, negative_slope=0.1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second block
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.leaky_relu(h, negative_slope=0.1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Third block with residual connection
        h = self.fc3(h)
        h = self.bn3(h)
        h = F.leaky_relu(h + identity, negative_slope=0.1)  # Add residual connection
        
        # Output
        pred = self.output(h)
        
        return pred

class MoleculeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, n_tasks=1, gnn_type='GCN', dropout=0.2):
        super(MoleculeGNN, self).__init__()
        
        # The GNN encoder now outputs a vector of dimension 'latent_dim'
        self.encoder = GNNEncoder(input_dim, hidden_dim, latent_dim, gnn_type, dropout=dropout)
        
        # The predictor now expects 'latent_dim' as its input dimension, not latent_dim*3
        self.predictor = PropertyPredictor(latent_dim, hidden_dim, n_tasks, dropout=dropout)
    
    def forward(self, data):
        z = self.encoder(data)
        pred = self.predictor(z)
        return pred
    
    def encode(self, data):
        return self.encoder(data)
