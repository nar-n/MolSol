"""
Feature Importance Analysis for GNNSol
This module provides functionality to analyze and export the importance of 
both graph-based and physicochemical features in the GNN solubility model.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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
    
    # PhysChem feature names
    physchem_feature_names = [
        "LogP", "TPSA", "H_Donors", "H_Acceptors", 
        "Molecular_Weight", "Rotatable_Bonds", "Aromatic_Rings", "Aliphatic_Rings"
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
                
                # Add physicochemical features if they exist
                if hasattr(data, 'physchem'):
                    modified_data.physchem = data.physchem.clone()
                
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
    
    # 2. Analyze importance of physicochemical descriptors
    # Check if the data has physicochemical descriptors
    if hasattr(X[0], 'physchem') and X[0].physchem is not None:
        physchem_dim = X[0].physchem.shape[1] 
        
        # New approach: Group-based perturbation for physicochemical features
        if hasattr(X[0], 'physchem') and X[0].physchem is not None:
            # 1. Analyze groups of related physicochemical features
            physchem_groups = {
                'Lipophilicity': [0],  # LogP
                'Polarity': [1],  # TPSA
                'H-Bonding': [2, 3],  # H-donors, H-acceptors
                'Molecular Size': [4, 5],  # MW, rotatable bonds
                'Ring Systems': [6, 7]  # Aromatic and aliphatic rings
            }
            
            # 2. Calculate importance for each group (synergistic effects)
            for group_name, feature_indices in physchem_groups.items():
                feature_preds = []
                category = "PhysChem Group"
                
                # Calculate mean values for all indices in this group
                group_means = []
                for pc_idx in feature_indices:
                    values = []
                    for data in X:
                        if hasattr(data, 'physchem') and data.physchem is not None:
                            values.append(data.physchem[0, pc_idx].item())
                    group_means.append(np.mean(values))
                
                # Make predictions with this group masked
                with torch.no_grad():
                    for idx, (data, _) in enumerate(zip(X, y)):
                        if not hasattr(data, 'physchem') or data.physchem is None:
                            continue
                            
                        # Create modified copy
                        modified_data = Data(
                            x=data.x.clone(),
                            edge_index=data.edge_index.clone(),
                            edge_attr=data.edge_attr.clone() if hasattr(data, 'edge_attr') else None,
                            physchem=data.physchem.clone(),
                            y=data.y.clone() if hasattr(data, 'y') else None
                        )
                        
                        # Modify all features in this group
                        for i, pc_idx in enumerate(feature_indices):
                            modified_data.physchem[0, pc_idx] = group_means[i]
                        
                        # Make prediction
                        pred = model(modified_data)
                        feature_preds.append(pred.item())
                
                # Calculate performance impact
                if feature_preds:
                    group_rmse = np.sqrt(mean_squared_error(y, feature_preds))
                    importance_score = group_rmse - baseline_rmse
                    
                    all_features.append(f"{group_name}_Group")
                    all_scores.append(importance_score)
                    all_categories.append(category)
            
            # 3. Analyze individual physicochemical descriptors as before
            for pc_idx in range(physchem_dim):
                feature_preds = []
                feature_name = physchem_feature_names[pc_idx] if pc_idx < len(physchem_feature_names) else f"PhysChem_{pc_idx}"
                category = "Physicochemical"
                
                # Calculate mean value for this physicochemical feature
                pc_values = []
                for data in X:
                    if hasattr(data, 'physchem') and data.physchem is not None:
                        pc_values.append(data.physchem[0, pc_idx].item())
                
                pc_mean = np.mean(pc_values)
                
                # Make predictions with this physicochemical feature masked
                with torch.no_grad():
                    for idx, (data, _) in enumerate(zip(X, y)):
                        # Skip if no physchem features
                        if not hasattr(data, 'physchem') or data.physchem is None:
                            continue
                            
                        # Create a modified copy of the data
                        modified_data = Data(
                            x=data.x.clone(),
                            edge_index=data.edge_index.clone(),
                            edge_attr=data.edge_attr.clone() if hasattr(data, 'edge_attr') else None,
                            physchem=data.physchem.clone(),
                            y=data.y.clone() if hasattr(data, 'y') else None
                        )
                        
                        # Modify the specific physicochemical feature
                        modified_data.physchem[0, pc_idx] = pc_mean
                        
                        # Make prediction
                        pred = model(modified_data)
                        feature_preds.append(pred.item())
                
                # Calculate performance impact
                if feature_preds:  # Only calculate if we have predictions
                    feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
                    importance_score = feature_rmse - baseline_rmse
                    
                    all_features.append(feature_name)
                    all_scores.append(importance_score)
                    all_categories.append(category)
        
        # Add cross-feature interaction analysis for key atom-physchem pairs
        if hasattr(X[0], 'physchem') and X[0].physchem is not None:
            # Analyze interactions between atom types and physiochemical properties
            key_atom_indices = [1, 2, 3]  # C, N, O atoms
            key_physchem_indices = [0, 1]  # LogP, TPSA
            
            for atom_idx in key_atom_indices:
                for pc_idx in key_physchem_indices:
                    feature_preds = []
                    atom_name = atom_feature_names[atom_idx]
                    pc_name = physchem_feature_names[pc_idx]
                    feature_name = f"{atom_name}+{pc_name}"
                    category = "Interaction"
                    
                    # Calculate mean values
                    atom_values = []
                    pc_values = []
                    
                    for data in X:
                        atom_values.extend(data.x[:, atom_idx].tolist())
                        if hasattr(data, 'physchem') and data.physchem is not None:
                            pc_values.append(data.physchem[0, pc_idx].item())
                    
                    atom_mean = np.mean(atom_values)
                    pc_mean = np.mean(pc_values)
                    
                    # Make predictions with both features masked
                    with torch.no_grad():
                        for idx, (data, _) in enumerate(zip(X, y)):
                            if not hasattr(data, 'physchem') or data.physchem is None:
                                continue
                                
                            # Create modified copy
                            modified_data = Data(
                                x=data.x.clone(),
                                edge_index=data.edge_index.clone(),
                                edge_attr=data.edge_attr.clone() if hasattr(data, 'edge_attr') else None,
                                physchem=data.physchem.clone(),
                                y=data.y.clone() if hasattr(data, 'y') else None
                            )
                            
                            # Modify both features
                            modified_data.x[:, atom_idx] = atom_mean
                            modified_data.physchem[0, pc_idx] = pc_mean
                            
                            # Make prediction
                            pred = model(modified_data)
                            feature_preds.append(pred.item())
                    
                    # Calculate interaction importance
                    if feature_preds:
                        interaction_rmse = np.sqrt(mean_squared_error(y, feature_preds))
                        
                        # Get individual importance scores
                        atom_score = next((score for i, (feat, score) in enumerate(zip(all_features, all_scores)) 
                                         if feat == atom_name), 0)
                        pc_score = next((score for i, (feat, score) in enumerate(zip(all_features, all_scores))
                                       if feat == pc_name), 0)
                        
                        # Calculate synergy score (interaction effect beyond individual effects)
                        synergy_score = interaction_rmse - baseline_rmse - atom_score - pc_score
                        
                        # Only add if synergy is significant
                        if abs(synergy_score) > 0.01:  # threshold
                            all_features.append(feature_name)
                            all_scores.append(synergy_score)
                            all_categories.append(category)
    
    # 3. Analyze edge features (bond types)
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
                    
                    # Add physicochemical features if they exist
                    if hasattr(data, 'physchem'):
                        modified_data.physchem = data.physchem.clone()
                    
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
    
    # 4. Analyze graph structure by removing edges
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
                
                # Add physicochemical features if they exist
                if hasattr(data, 'physchem'):
                    modified_data.physchem = data.physchem.clone()
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
    
    # Plot feature importance
    plot_feature_importance(df, output_dir)
    
    return df

def plot_feature_importance(df, output_dir):
    """
    Create feature importance plot and save it as PNG
    
    Args:
        df: DataFrame with feature importance results
        output_dir: Directory where to save the plot
    """
    # Use top N features for clarity
    top_n = min(15, len(df))
    top_df = df.head(top_n).copy()
    
    # Sort by importance for the plot (ascending for bottom-to-top display)
    top_df = top_df.sort_values('Importance_Score', ascending=True)
    
    # Create color map based on categories
    category_colors = {
        'Atomic': '#1f77b4',         # Blue
        'Atom Property': '#ff7f0e',  # Orange
        'Bond Type': '#2ca02c',      # Green
        'Graph Structure': '#d62728', # Red
        'Physicochemical': '#9467bd',  # Purple
        'PhysChem Group': '#e377c2',  # Pink
        'Interaction': '#17becf'     # Cyan
    }
    
    # Create default color for any other categories
    default_color = '#8c564b'  # Brown
    
    # Create list of colors for the bars
    colors = [category_colors.get(cat, default_color) for cat in top_df['Category']]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_df['Feature'], top_df['Importance_Score'], color=colors)
    
    # Add feature importance values as text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.01,            # Slightly to the right of the bar
            bar.get_y() + bar.get_height()/2,  # Vertically centered
            f'{width:.3f}',          # Format with 3 decimal places
            va='center'              # Vertically centered
        )
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # Create legend based on unique categories
    unique_categories = top_df['Category'].unique()
    legend_handles = [plt.Rectangle((0,0),1,1, color=category_colors.get(cat, default_color)) for cat in unique_categories]
    plt.legend(legend_handles, unique_categories, loc='lower right')
    
    plt.title('Feature Importance for Solubility Prediction', fontsize=14)
    plt.xlabel('Importance Score (Change in RMSE)', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'feature_importance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {plot_path}")
    
    # Create a separate visualization for physicochemical features to highlight their contribution
    physchem_df = df[df['Category'].isin(['Physicochemical', 'PhysChem Group', 'Interaction'])]
    
    if not physchem_df.empty:
        plt.figure(figsize=(12, 6))
        
        # Sort by importance
        physchem_df = physchem_df.sort_values('Importance_Score', ascending=True)
        
        # Color map for different categories
        category_colors = {
            'Physicochemical': '#9467bd',
            'PhysChem Group': '#e377c2',
            'Interaction': '#17becf'
        }
        
        colors = [category_colors.get(cat, '#bcbd22') for cat in physchem_df['Category']]
        
        # Create bar chart
        bars = plt.barh(physchem_df['Feature'], physchem_df['Importance_Score'], color=colors)
        
        # Add feature importance values as text labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.005,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                va='center'
            )
        
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Create legend
        unique_categories = physchem_df['Category'].unique()
        legend_handles = [plt.Rectangle((0,0),1,1, color=category_colors.get(cat, '#bcbd22'))
                        for cat in unique_categories]
        plt.legend(legend_handles, unique_categories, loc='best')
        
        plt.title('Physicochemical Feature Importance (Including Interactions)', fontsize=14)
        plt.xlabel('Importance Score (Change in RMSE)', fontsize=12)
        plt.tight_layout()
        
        # Save physchem-focused plot
        pc_plot_path = os.path.join(output_dir, 'physchem_importance_plot.png')
        plt.savefig(pc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Physicochemical feature importance plot saved to {pc_plot_path}")

# Add a new function to calculate the normalized contribution of feature types
def analyze_feature_contribution_by_type(df):
    """
    Analyze overall contribution by feature type
    """
    # Group by category and sum absolute importance scores
    category_contrib = df.groupby('Category')['Importance_Score'].apply(lambda x: sum(abs(v) for v in x)).reset_index()
    
    # Calculate percentage contribution
    total_importance = category_contrib['Importance_Score'].sum()
    category_contrib['Percentage'] = category_contrib['Importance_Score'] / total_importance * 100
    
    # Sort by importance
    category_contrib = category_contrib.sort_values('Importance_Score', ascending=False)
    
    return category_contrib
