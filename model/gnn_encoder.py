import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
from model.batch_norm_utils import SingleBatchNorm1d

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, gnn_type='GCN', dropout=0.2):
        super(GNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        
        # Select GNN layer type with increased depth
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim)  # Added deeper layer
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.conv4 = GATConv(hidden_dim*4, hidden_dim, heads=1, dropout=dropout)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Improved projection to latent space - use our custom SingleBatchNorm1d
        self.project = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*2),  # From concatenated features
            SingleBatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, latent_dim),  # Final output size is latent_dim
            SingleBatchNorm1d(latent_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create a dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GNN layers with residual connections and stronger feature extraction
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index)) + x1  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = F.relu(self.conv3(x2, edge_index)) + x2  # Residual connection
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        x4 = F.relu(self.conv4(x3, edge_index)) + x3  # Residual connection
        
        # Multiple pooling strategies for richer graph representation
        pool_mean = global_mean_pool(x4, batch)
        pool_max = global_max_pool(x4, batch)
        pool_sum = global_add_pool(x4, batch)
        
        # Combine different pooling methods
        pooled = torch.cat([pool_mean, pool_max, pool_sum], dim=1)
        
        # Project to latent space, with exactly latent_dim output dimensions
        z = self.project(pooled)
        
        return z
