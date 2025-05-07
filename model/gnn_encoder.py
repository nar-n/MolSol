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
        self.physchem_dim = 8  # Expected dimension of physicochemical features
        
        # Select GNN layer type with increased depth
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.conv4 = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_dim*4, hidden_dim, heads=4, dropout=dropout)
            self.conv4 = GATConv(hidden_dim*4, hidden_dim, heads=1, dropout=dropout)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Normalize physicochemical features early in the network
        self.physchem_norm = nn.LayerNorm(self.physchem_dim)
        
        # Create a direct path for physicochemical features
        self.physchem_encoder = nn.Sequential(
            nn.Linear(self.physchem_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Improved projection to latent space that combines graph and physicochemical features
        self.project = nn.Sequential(
            nn.Linear(hidden_dim*3 + hidden_dim, hidden_dim*2),  # Added physchem features
            SingleBatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, latent_dim),
            SingleBatchNorm1d(latent_dim)
        )
        
        # Gate mechanism to dynamically weight the contribution of physicochemical vs. graph features
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim*3 + hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Process physicochemical features if available
        has_physchem = hasattr(data, 'physchem') and data.physchem is not None
        if has_physchem:
            physchem_data = data.physchem
            # Normalize physicochemical features
            physchem_data = self.physchem_norm(physchem_data)
            # Encode physicochemical features
            physchem_encoded = self.physchem_encoder(physchem_data)
        else:
            # Create zero tensor as placeholder
            physchem_encoded = torch.zeros(batch.max().item() + 1, self.physchem_encoder[-2].out_features, 
                                          device=x.device)
        
        # If batch is None (single graph), create a dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GNN layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index)) + x1
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = F.relu(self.conv3(x2, edge_index)) + x2
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        
        x4 = F.relu(self.conv4(x3, edge_index)) + x3
        
        # Multiple pooling strategies for graph features
        pool_mean = global_mean_pool(x4, batch)
        pool_max = global_max_pool(x4, batch)
        pool_sum = global_add_pool(x4, batch)
        
        # Combine different pooling methods for graph features
        pooled_graph = torch.cat([pool_mean, pool_max, pool_sum], dim=1)
        
        # Combine graph features with physicochemical features
        if has_physchem:
            # Expand physchem features if needed (for batch size > 1)
            if physchem_encoded.shape[0] == 1 and pooled_graph.shape[0] > 1:
                physchem_encoded = physchem_encoded.expand(pooled_graph.shape[0], -1)
            
            # Combine features using gate mechanism for dynamic weighting
            combined = torch.cat([pooled_graph, physchem_encoded], dim=1)
            gate_weight = self.feature_gate(combined)
            
            # Weight the features (graph features vs physchem features)
            weighted_pooled = gate_weight * pooled_graph
            weighted_physchem = (1 - gate_weight) * physchem_encoded
            
            # Concatenate for final projection
            combined_features = torch.cat([weighted_pooled, weighted_physchem], dim=1)
            z = self.project(combined_features)
        else:
            # If no physchem features, just use graph features with zeros for physchem
            combined = torch.cat([pooled_graph, physchem_encoded], dim=1)
            z = self.project(combined)
        
        return z
