"""
Feature Importance Analysis for GNNSol
This module provides functionality to analyze and export the importance of 
both graph-based and physicochemical features in the GNN solubility model.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, Batch

def export_feature_importance_to_csv(model, X, y, output_dir):
    """
    Analyze and export the importance of all features to a CSV file
    
    Args:
        model: The trained GNN model
        X: List of PyG Data objects containing molecular graph representations
        y: List of target solubility values
        output_dir: Directory where to save the CSV file
    """
    print("\nAnalyzing importance of all features...")
    
    # Baseline prediction with full model
    model.eval()
    baseline_preds = []
    
    with torch.no_grad():
        for data, _ in zip(X, y):
            pred = model(data)
            baseline_preds.append(pred.item())
    
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    # Create feature names
    # 1. Graph-based node features - assuming first 10 features are atomic numbers
    # This should match the node feature encoding in SMILESToGraph
    atom_feature_names = [
        "Atomic_H", "Atomic_C", "Atomic_N", "Atomic_O", "Atomic_F", 
        "Atomic_P", "Atomic_S", "Atomic_Cl", "Atomic_Br", "Atomic_I"
    ]
    other_node_feature_names = [
        "Formal_Charge", "Radical_Electrons", "Aromaticity", "Num_Hydrogen_Atoms"
    ]
    
    # All feature importance results
    all_features = []
    all_scores = []
    all_categories = []
    
    # 1. Analyze importance of graph-based node features
    # For each node feature dimension
    for feature_idx in range(X[0].x.shape[1]):
        feature_preds = []
        
        # Get the appropriate feature name
        if feature_idx < 10:
            feature_name = atom_feature_names[feature_idx]
            category = "Atomic"
        else:
            feature_name = other_node_feature_names[feature_idx - 10]
            category = "Atom Property"
            
        # Calculate mean values for this feature across all molecules and atoms
        feature_values = []
        for data in X:
            # Get values for this feature across all atoms
            feature_values.extend(data.x[:, feature_idx].tolist())
        
        feature_mean = np.mean(feature_values)
        
        # Make predictions with this feature masked
        with torch.no_grad():
            for idx, (data, _) in enumerate(zip(X, y)):
                # Create a modified copy of the data
                modified_data = Data(
                    x=data.x.clone(),
                    edge_index=data.edge_index.clone(),
                    edge_attr=data.edge_attr.clone() if hasattr(data, 'edge_attr') else None,
                    y=data.y.clone() if hasattr(data, 'y') else None
                )
                
                # Modify the specific feature
                modified_data.x[:, feature_idx] = feature_mean
                
                # Make prediction
                pred = model(modified_data)
                feature_preds.append(pred.item())
        
        # Calculate performance impact
        feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
        importance_score = feature_rmse - baseline_rmse
        
        all_features.append(feature_name)
        all_scores.append(importance_score)
        all_categories.append(category)
    
    # 2. Analyze edge features (bond types)
    if hasattr(X[0], 'edge_attr') and X[0].edge_attr is not None and X[0].edge_attr.shape[1] > 0:
        bond_feature_names = ["Single", "Double", "Triple", "Aromatic"]
        
        for bond_idx in range(X[0].edge_attr.shape[1]):
            feature_preds = []
            feature_name = f"Bond_{bond_feature_names[bond_idx]}"
            category = "Bond Type"
            
            # Calculate mean value for this bond feature
            bond_values = []
            for data in X:
                if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
                    bond_values.extend(data.edge_attr[:, bond_idx].tolist())
            
            bond_mean = np.mean(bond_values) if bond_values else 0.0
            
            # Make predictions with this bond feature masked
            with torch.no_grad():
                for idx, (data, _) in enumerate(zip(X, y)):
                    # Create a modified copy of the data
                    modified_data = Data(
                        x=data.x.clone(),
                        edge_index=data.edge_index.clone(),
                        edge_attr=data.edge_attr.clone() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
                        y=data.y.clone() if hasattr(data, 'y') else None
                    )
                    
                    # Modify the specific bond feature if available
                    if hasattr(modified_data, 'edge_attr') and modified_data.edge_attr is not None and modified_data.edge_attr.shape[0] > 0:
                        modified_data.edge_attr[:, bond_idx] = bond_mean
                    
                    # Make prediction
                    pred = model(modified_data)
                    feature_preds.append(pred.item())
            
            # Calculate performance impact
            feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
            importance_score = feature_rmse - baseline_rmse
            
            all_features.append(feature_name)
            all_scores.append(importance_score)
            all_categories.append(category)
    
    # 3. Analyze graph structure by removing edges
    # Make predictions with 50% of random edges removed
    structure_preds = []
    with torch.no_grad():
        for idx, (data, _) in enumerate(zip(X, y)):
            # Create a modified copy with 50% of edges removed randomly
            if data.edge_index.shape[1] > 0:  # Check if there are edges
                num_edges = data.edge_index.shape[1] // 2  # Each edge appears twice (undirected)
                keep_edges = torch.randperm(num_edges)[:num_edges//2]  # Keep 50%
                
                # Get the edges to keep (both directions for undirected graph)
                edge_indices_to_keep = torch.cat([keep_edges * 2, keep_edges * 2 + 1])
                
                modified_data = Data(
                    x=data.x.clone(),
                    edge_index=data.edge_index[:, edge_indices_to_keep],
                    edge_attr=data.edge_attr[edge_indices_to_keep].clone() if hasattr(data, 'edge_attr') and data.edge_attr is not None else None,
                    y=data.y.clone() if hasattr(data, 'y') else None
                )
            else:
                modified_data = data
                
            # Make prediction
            pred = model(modified_data)
            structure_preds.append(pred.item())
    
    # Calculate performance impact of graph structure
    structure_rmse = np.sqrt(mean_squared_error(y, structure_preds))
    structure_importance = structure_rmse - baseline_rmse
    
    all_features.append("Graph_Structure")
    all_scores.append(structure_importance)
    all_categories.append("Graph Structure")
    
    # Create a dataframe with all results
    df = pd.DataFrame({
        'Feature': all_features,
        'Category': all_categories,
        'Importance_Score': all_scores
    })
    
    # Sort by importance score (highest first)
    df = df.sort_values('Importance_Score', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    df.to_csv(csv_path, index=False)
    print(f"Feature importance scores saved to {csv_path}")
    
    return df
