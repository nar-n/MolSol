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

def export_feature_importance_to_csv(model, X, y, output_dir):
    """
    Analyze and export the importance of all features to a CSV file
    
    Args:
        model: The trained GNN model
        X: List of tuples (node_features, adj_matrix, physchem_features, mol)
        y: List of target solubility values
        output_dir: Directory where to save the CSV file
    """
    print("\nAnalyzing importance of all features...")
    
    # Baseline prediction with full model
    model.eval()
    baseline_preds = []
    
    with torch.no_grad():
        for features_adj_physchem_mol, _ in zip(X, y):
            features, adj, physchem, _ = features_adj_physchem_mol
            _, pred = model(features, adj, physchem)
            baseline_preds.append(pred.item())
    
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    # Create feature names
    # 1. Graph-based node features
    atom_feature_names = [
        "Atomic_H", "Atomic_C", "Atomic_N", "Atomic_O", "Atomic_F", 
        "Atomic_P", "Atomic_S", "Atomic_Cl", "Atomic_Br", "Atomic_I"
    ]
    other_node_feature_names = [
        "Formal_Charge", "Aromaticity", "Atom_Degree", "Num_Hydrogen_Atoms"
    ]
    
    # 2. Physicochemical features
    physchem_feature_names = [
        "Molecular_Weight", "LogP", "H-bond_Donors", "H-bond_Acceptors", 
        "TPSA", "Rotatable_Bonds", "Rings", "Fraction_sp3", 
        "Aromatic_Rings", "Heteroatoms", "Molar_Refractivity", 
        "Labute_ASA", "QED"
    ]
    
    # All feature importance results
    all_features = []
    all_scores = []
    all_categories = []
    
    # 1. Analyze importance of graph-based node features
    # This requires a more complex approach since we need to mask specific
    # node features across all atoms in a molecule
    
    # Get the index ranges for different node feature types
    atom_indices = list(range(10))  # First 10 features are atomic numbers
    other_indices = list(range(10, 14))  # Next 4 are other atom properties
    
    # For each graph-based feature type
    for feature_idx in range(14):
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
        for x_item in X:
            node_features = x_item[0]
            # Get values for this feature across all atoms
            feature_values.extend(node_features[:, feature_idx].tolist())
        
        feature_mean = np.mean(feature_values)
        
        # Make predictions with this feature masked
        with torch.no_grad():
            for idx, (features_adj_physchem_mol, _) in enumerate(zip(X, y)):
                features, adj, physchem, _ = features_adj_physchem_mol
                
                # Create a modified copy of the features tensor
                modified_features = features.clone()
                modified_features[:, feature_idx] = feature_mean
                
                _, pred = model(modified_features, adj, physchem)
                feature_preds.append(pred.item())
        
        # Calculate performance impact
        feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
        importance_score = feature_rmse - baseline_rmse
        
        all_features.append(feature_name)
        all_scores.append(importance_score)
        all_categories.append(category)
    
    # 2. Analyze importance of physicochemical features
    for feature_idx in range(len(physchem_feature_names)):
        feature_preds = []
        feature_name = physchem_feature_names[feature_idx]
        category = "Physicochemical"
        
        # Calculate mean values for this physicochemical feature
        feature_values = [X[i][2][0, feature_idx].item() for i in range(len(X))]
        feature_mean = np.mean(feature_values)
        
        # Make predictions with this feature masked
        with torch.no_grad():
            for idx, (features_adj_physchem_mol, _) in enumerate(zip(X, y)):
                features, adj, physchem, _ = features_adj_physchem_mol
                
                # Create modified copy with this feature changed to mean
                modified_physchem = physchem.clone()
                modified_physchem[0, feature_idx] = feature_mean
                
                _, pred = model(features, adj, modified_physchem)
                feature_preds.append(pred.item())
        
        # Calculate performance impact
        feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
        importance_score = feature_rmse - baseline_rmse
        
        all_features.append(feature_name)
        all_scores.append(importance_score)
        all_categories.append(category)
    
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
