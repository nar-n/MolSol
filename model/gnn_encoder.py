import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, gnn_type='GCN'):
        super(GNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Select GNN layer type
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
            self.conv3 = GATConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Projection to latent space
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create a dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GNN layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        
        # Combine features from different layers
        x = x1 + x2 + x3
        
        # Global pooling (mean of node features)
        pooled = global_mean_pool(x, batch)
        
        # Project to latent space
        z = self.project(pooled)
        
        return z
