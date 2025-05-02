import torch
import torch.nn as nn

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, n_tasks=1):
        super(PropertyPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_tasks)
        )
    
    def forward(self, z):
        return self.predictor(z)

class MoleculeGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, n_tasks=1, gnn_type='GCN'):
        super(MoleculeGNN, self).__init__()
        
        self.encoder = GNNEncoder(input_dim, hidden_dim, latent_dim, gnn_type)
        self.predictor = PropertyPredictor(latent_dim, hidden_dim, n_tasks)
    
    def forward(self, data):
        z = self.encoder(data)
        pred = self.predictor(z)
        return pred
    
    def encode(self, data):
        return self.encoder(data)
