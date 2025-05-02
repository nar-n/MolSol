import torch
import pandas as pd
from torch_geometric.data import Dataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.smiles_to_graph import SMILESToGraph

class MoleculeDataset(Dataset):
    def __init__(self, csv_file, smiles_col, target_cols, transform=None):
        super(MoleculeDataset, self).__init__()
        
        self.data = pd.read_csv(csv_file)
        self.smiles_col = smiles_col
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.transform = transform
        self.converter = SMILESToGraph()
        
        # Process all SMILES to graphs
        valid_indices = []
        self.processed_data = []
        
        for idx, row in self.data.iterrows():
            smiles = row[self.smiles_col]
            targets = [float(row[col]) for col in self.target_cols]
            
            graph_data = self.converter.convert(smiles)
            if graph_data is not None:
                graph_data.y = torch.tensor(targets, dtype=torch.float)
                self.processed_data.append(graph_data)
                valid_indices.append(idx)
        
        print(f"Successfully processed {len(self.processed_data)} out of {len(self.data)} molecules")
    
    def len(self):
        return len(self.processed_data)
    
    def get(self, idx):
        data = self.processed_data[idx]
        if self.transform:
            data = self.transform(data)
        return data
