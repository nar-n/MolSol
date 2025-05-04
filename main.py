"""
Streamlined GNN for Molecular Solubility Prediction
This script focuses only on:
1. Loading ESOL dataset
2. Keeping a percentage of molecules as unseen prediction set
3. Training and testing the model
4. Evaluating performance
5. Keeping the best model
6. Predicting solubility of unseen molecules
7. Reporting results
8. Advanced model evaluation techniques
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, rdMolDescriptors
# Import QED properly or provide alternative
try:
    from rdkit.Chem import QED  # In newer versions of RDKit, QED is a separate module
except ImportError:
    QED = None
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import time
import datetime
import os
import copy
from collections import defaultdict

# %% Molecular Graph Construction
class MoleculeGraphConverter:
    def __init__(self):
        # Atom features: Atomic number, hybridization, aromaticity, num H, formal charge
        self.atom_features = {
            'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35, 53],  # H, C, N, O, F, P, S, Cl, Br, I
        }
        
    def calculate_physchem_descriptors(self, mol):
        """Calculate physicochemical descriptors for a molecule"""
        descriptors = []
        
        try:
            # Calculate basic molecular properties
            descriptors.append(Descriptors.MolWt(mol))            # Molecular Weight
            descriptors.append(Descriptors.MolLogP(mol))           # LogP
            descriptors.append(Lipinski.NumHDonors(mol))           # H-bond donors
            descriptors.append(Lipinski.NumHAcceptors(mol))        # H-bond acceptors
            descriptors.append(rdMolDescriptors.CalcTPSA(mol))     # Topological Polar Surface Area
            descriptors.append(Lipinski.NumRotatableBonds(mol))    # Number of rotatable bonds
            descriptors.append(rdMolDescriptors.CalcNumRings(mol)) # Number of rings
            
            # Add more complex descriptors
            descriptors.append(rdMolDescriptors.CalcFractionCSP3(mol))  # Fraction of sp3 carbons
            descriptors.append(Descriptors.NumAromaticRings(mol))       # Number of aromatic rings  
            descriptors.append(Chem.rdMolDescriptors.CalcNumHeteroatoms(mol)) # Number of heteroatoms
            
            # Water solubility-related descriptors
            descriptors.append(Chem.Crippen.MolMR(mol))            # Molar refractivity
            descriptors.append(Descriptors.LabuteASA(mol))         # Labute Accessible Surface Area
            
            # Drug-likeness - Handle QED differently based on availability
            if QED is not None:
                descriptors.append(QED.qed(mol))           # QED drug-likeness score
            else:
                # Alternative: use a fixed value or omit this feature
                descriptors.append(0.5)  # Default mid-range value
        
        except Exception as e:
            print(f"Warning: Error calculating descriptors for molecule: {e}")
            # Return a vector of zeros with the appropriate length
            # 12 features (13 with QED)
            return torch.zeros(1, 13, dtype=torch.float)
        
        return torch.tensor(descriptors, dtype=torch.float).reshape(1, -1)
        
    def smiles_to_graph(self, smiles):
        """Convert SMILES string to molecular graph with node and edge features"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Create graph structure
        num_atoms = mol.GetNumAtoms()
        adj_matrix = torch.zeros(num_atoms, num_atoms)
        
        # Add bonds (edges)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adj_matrix[i, j] = bond_type
            adj_matrix[j, i] = bond_type  # Undirected graph
        
        # Add node features
        node_features = []
        for atom in mol.GetAtoms():
            features = []
            # Atomic number one-hot encoding
            atom_num = atom.GetAtomicNum()
            atom_feat = [int(atom_num == num) for num in self.atom_features['atomic_num']]
            features.extend(atom_feat)
            
            # Additional atom features
            features.append(atom.GetFormalCharge())
            features.append(int(atom.GetIsAromatic()))
            features.append(atom.GetDegree())
            features.append(atom.GetTotalNumHs())
            
            node_features.append(features)
        
        # Calculate physicochemical descriptors
        physchem_descriptors = self.calculate_physchem_descriptors(mol)
        
        return {
            'mol': mol,
            'adj_matrix': adj_matrix,
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'physchem_features': physchem_descriptors,
            'smiles': smiles
        }

# %% Advanced Evaluation Techniques

class ModelEvaluator:
    """Class for advanced model evaluation techniques"""
    
    @staticmethod
    def calculate_extended_metrics(y_true, y_pred):
        """Calculate extended set of evaluation metrics including Pearson, Spearman, Q², and MAPE"""
        metrics = {}
        
        # Convert lists to numpy arrays if they're not already
        y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
        y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
        
        # Standard metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional statistical metrics
        # Pearson correlation
        pearson_corr, p_value = pearsonr(y_true, y_pred)
        metrics['pearson_r'] = pearson_corr
        metrics['pearson_p'] = p_value
        
        # Spearman correlation
        spearman_corr, s_p_value = spearmanr(y_true, y_pred)
        metrics['spearman_rho'] = spearman_corr
        metrics['spearman_p'] = s_p_value
        
        # Q² (predictive R²) - implemented as a variant of R² based on predicted values
        # This is an approximation of Q² for demonstration purposes
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        press = np.sum((y_true - y_pred) ** 2)  # PRESS: prediction error sum of squares
        metrics['q2'] = 1 - (press / ss_total)
        
        # MAPE (Mean Absolute Percentage Error)
        # Handle potential division by zero or very small values
        abs_percentage_errors = []
        for true, pred in zip(y_true, y_pred):
            if abs(true) > 0.001:  # Avoid division by very small numbers
                abs_percentage_errors.append(abs((true - pred) / true) * 100)
        
        if abs_percentage_errors:
            metrics['mape'] = np.mean(abs_percentage_errors)
        else:
            metrics['mape'] = float('nan')
            
        return metrics
    
    @staticmethod
    def perform_kfold_cv(X, y, k=10, epochs=100, batch_size=32):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_metrics = []
        all_predictions = []
        all_targets = []
        
        print(f"\nPerforming {k}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(X)))):
            print(f"Fold {fold+1}/{k}...")
            
            # Split data
            X_train_fold = [X[i] for i in train_idx]
            y_train_fold = [y[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_val_fold = [y[i] for i in val_idx]
            
            # Create model
            feature_dim = X_train_fold[0][0].shape[1]  # Node features dimension
            physchem_dim = X_train_fold[0][2].shape[1]  # Physicochemical features dimension
            model = MolecularGNN(in_features=feature_dim, hidden_features=128, latent_dim=64, 
                             physchem_features=physchem_dim, out_features=1)
            
            # Train model
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            loss_fn = torch.nn.MSELoss()
            
            # Simple training loop for CV
            for epoch in range(epochs):
                # Training
                model.train()
                for i in range(0, len(X_train_fold), batch_size):
                    batch_indices = range(i, min(i+batch_size, len(X_train_fold)))
                    
                    for idx in batch_indices:
                        features, adj, physchem, _ = X_train_fold[idx]
                        target = y_train_fold[idx]
                        
                        optimizer.zero_grad()
                        _, pred = model(features, adj, physchem)
                        loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                        loss.backward()
                        optimizer.step()
                
                # Print progress at specific intervals
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}")
            
            # Evaluate on validation set
            model.eval()
            fold_preds = []
            
            with torch.no_grad():
                for features_adj_physchem_mol, target in zip(X_val_fold, y_val_fold):
                    features, adj, physchem, _ = features_adj_physchem_mol
                    _, pred = model(features, adj, physchem)
                    fold_preds.append(pred.item())
                    
                    all_predictions.append(pred.item())
                    all_targets.append(target)
            
            # Calculate metrics for this fold
            fold_metric = ModelEvaluator.calculate_extended_metrics(y_val_fold, fold_preds)
            fold_metrics.append(fold_metric)
            
            print(f"  Fold {fold+1} metrics: RMSE={fold_metric['rmse']:.4f}, R²={fold_metric['r2']:.4f}")
        
        # Calculate average metrics across all folds
        avg_metrics = defaultdict(float)
        for metric in fold_metrics[0].keys():
            avg_metrics[metric] = sum(fold[metric] for fold in fold_metrics) / k
        
        # Calculate aggregated metrics on all predictions
        overall_metrics = ModelEvaluator.calculate_extended_metrics(all_targets, all_predictions)
        
        return {
            'fold_metrics': fold_metrics,
            'avg_metrics': dict(avg_metrics),
            'overall_metrics': overall_metrics
        }
    
    @staticmethod
    def analyze_feature_importance(model, X, y, output_dir=None):
        """Analyze feature importance by removing each feature and measuring performance impact"""
        print("\nAnalyzing feature importance...")
        
        # Baseline prediction with full model
        model.eval()
        baseline_preds = []
        
        with torch.no_grad():
            for features_adj_physchem_mol, _ in zip(X, y):
                features, adj, physchem, _ = features_adj_physchem_mol
                _, pred = model(features, adj, physchem)
                baseline_preds.append(pred.item())
        
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))
        
        # Feature names for physicochemical features
        physchem_feature_names = [
            "Molecular Weight", "LogP", "H-bond Donors", "H-bond Acceptors", 
            "TPSA", "Rotatable Bonds", "Rings", "Fraction sp3", 
            "Aromatic Rings", "Heteroatoms", "Molar Refractivity", 
            "Labute ASA", "QED"
        ]
        
        # Analyze importance of physicochemical features
        importance_scores = []
        
        for feature_idx in range(X[0][2].shape[1]):
            # Skip specific features for this run by setting to mean
            feature_preds = []
            feature_values = [X[i][2][0, feature_idx].item() for i in range(len(X))]
            feature_mean = np.mean(feature_values)
            
            with torch.no_grad():
                for idx, (features_adj_physchem_mol, _) in enumerate(zip(X, y)):
                    features, adj, physchem, _ = features_adj_physchem_mol
                    
                    # Create modified copy of physchem with one feature zeroed out
                    modified_physchem = physchem.clone()
                    modified_physchem[0, feature_idx] = feature_mean
                    
                    _, pred = model(features, adj, modified_physchem)
                    feature_preds.append(pred.item())
            
            # Calculate performance impact when this feature is removed
            feature_rmse = np.sqrt(mean_squared_error(y, feature_preds))
            importance_score = feature_rmse - baseline_rmse
            importance_scores.append(importance_score)
        
        # Create feature importance visualization
        plt.figure(figsize=(12, 8))
        sorted_indices = np.argsort(importance_scores)
        plt.barh([physchem_feature_names[i] for i in sorted_indices], 
                 [importance_scores[i] for i in sorted_indices])
        plt.xlabel('Increase in RMSE when feature is removed')
        plt.title('Physicochemical Feature Importance')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        else:
            plt.savefig('feature_importance.png')
        plt.close()
        
        # Return sorted feature importance
        return [(physchem_feature_names[i], importance_scores[i]) 
                for i in sorted_indices[::-1]]  # Reverse to get highest importance first

# %% GNN Model Definition
class MolecularGNN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, latent_dim, physchem_features, out_features):
        super(MolecularGNN, self).__init__()
        # Encoding layers for graph structure
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, latent_dim)
        
        # Layers for physicochemical features
        self.physchem_fc1 = torch.nn.Linear(physchem_features, hidden_features)
        self.physchem_fc2 = torch.nn.Linear(hidden_features, latent_dim)
        
        # Combined property prediction head
        self.fc_combined = torch.nn.Linear(latent_dim * 2, latent_dim)
        self.fc_out = torch.nn.Linear(latent_dim, out_features)
        
        # Activation functions
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, adj_matrix, physchem):
        # Graph pathway - Message passing layers (graph convolution)
        x = self.fc1(x)
        x = self.relu(torch.mm(adj_matrix, x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(torch.mm(adj_matrix, x))
        x = self.dropout(x)
        
        # Encoding to latent space
        graph_latent = self.fc3(x)
        
        # Global pooling (mean of node embeddings)
        graph_embedding = torch.mean(graph_latent, dim=0, keepdim=True)
        
        # Physicochemical pathway
        physchem_hidden = self.relu(self.physchem_fc1(physchem))
        physchem_hidden = self.dropout(physchem_hidden)
        physchem_latent = self.relu(self.physchem_fc2(physchem_hidden))
        
        # Combine both pathways
        combined = torch.cat([graph_embedding, physchem_latent], dim=1)
        combined = self.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        
        # Property prediction
        out = self.fc_out(combined)
        
        return graph_latent, out

# %% Feature Importance CSV Generation
def generate_feature_importance_csv(model, X_test, y_test, output_dir=None):
    """
    Generate a comprehensive CSV file showing the importance of all features
    including both graph-based (atom types, properties) and physicochemical descriptors.
    
    Args:
        model: Trained GNN model
        X_test: List of test data tuples (node_features, adj_matrix, physchem_features, mol)
        y_test: List of test labels
        output_dir: Directory to save the CSV and visualization
        
    Returns:
        DataFrame containing feature importance scores
    """
    print("\nGenerating comprehensive feature importance analysis...")
    feature_data = []
    
    # Get baseline prediction with full model
    model.eval()
    baseline_preds = []
    
    with torch.no_grad():
        for features_adj_physchem_mol, _ in zip(X_test, y_test):
            features, adj, physchem, _ = features_adj_physchem_mol
            _, pred = model(features, adj, physchem)
            baseline_preds.append(pred.item())
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    
    # 1. Analyze Atom Types (first 10 features in node_features are one-hot encoded atom types)
    print("Analyzing atom type features...")
    atom_types = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    
    for atom_idx, atom_type in enumerate(atom_types):
        atom_type_preds = []
        
        with torch.no_grad():
            for idx, (features_adj_physchem_mol, _) in enumerate(zip(X_test, y_test)):
                features, adj, physchem, _ = features_adj_physchem_mol
                
                # Create modified copy of features with atom type zeroed out
                modified_features = features.clone()
                modified_features[:, atom_idx] = 0.0
                
                _, pred = model(modified_features, adj, physchem)
                atom_type_preds.append(pred.item())
        
        # Calculate impact when this atom type is removed
        atom_rmse = np.sqrt(mean_squared_error(y_test, atom_type_preds))
        importance_score = atom_rmse - baseline_rmse
        
        # Add to feature data
        feature_data.append({
            'Feature': f'Atom: {atom_type}',
            'Category': 'Graph-Atom Type',
            'Importance_Score': importance_score
        })
    
    # 2. Analyze Atom Properties (formal charge, aromaticity, degree, num H)
    print("Analyzing atom property features...")
    atom_properties = ['Formal Charge', 'Aromaticity', 'Atom Degree', 'Number of H']
    
    for prop_idx, prop_name in enumerate(atom_properties):
        prop_preds = []
        
        with torch.no_grad():
            for idx, (features_adj_physchem_mol, _) in enumerate(zip(X_test, y_test)):
                features, adj, physchem, _ = features_adj_physchem_mol
                
                # Create modified copy with property zeroed out (starting at index 10)
                modified_features = features.clone()
                modified_features[:, 10 + prop_idx] = 0.0
                
                _, pred = model(modified_features, adj, physchem)
                prop_preds.append(pred.item())
        
        # Calculate impact when this property is removed
        prop_rmse = np.sqrt(mean_squared_error(y_test, prop_preds))
        importance_score = prop_rmse - baseline_rmse
        
        # Add to feature data
        feature_data.append({
            'Feature': prop_name,
            'Category': 'Graph-Atom Property',
            'Importance_Score': importance_score
        })
    
    # 3. Analyze Graph Structure (bond connectivity)
    print("Analyzing graph structural features...")
    
    # Effect of edge weights (reducing by 50%)
    edge_preds = []
    with torch.no_grad():
        for idx, (features_adj_physchem_mol, _) in enumerate(zip(X_test, y_test)):
            features, adj, physchem, _ = features_adj_physchem_mol
            
            # Reduce edge weights by 50%
            modified_adj = adj.clone() * 0.5
            
            _, pred = model(features, modified_adj, physchem)
            edge_preds.append(pred.item())
    
    edge_rmse = np.sqrt(mean_squared_error(y_test, edge_preds))
    importance_score = edge_rmse - baseline_rmse
    
    feature_data.append({
        'Feature': 'Bond Connectivity',
        'Category': 'Graph-Structure',
        'Importance_Score': importance_score
    })
    
    # Effect of removing edges (sparsify graph by keeping only strong bonds)
    sparse_preds = []
    with torch.no_grad():
        for idx, (features_adj_physchem_mol, _) in enumerate(zip(X_test, y_test)):
            features, adj, physchem, _ = features_adj_physchem_mol
            
            # Only keep bonds with weight > 1.0 (typically double and triple bonds)
            modified_adj = adj.clone()
            modified_adj[modified_adj <= 1.0] = 0.0
            
            _, pred = model(features, modified_adj, physchem)
            sparse_preds.append(pred.item())
    
    sparse_rmse = np.sqrt(mean_squared_error(y_test, sparse_preds))
    importance_score = sparse_rmse - baseline_rmse
    
    feature_data.append({
        'Feature': 'Strong Bonds Only',
        'Category': 'Graph-Structure',
        'Importance_Score': importance_score
    })
    
    # 4. Analyze Physicochemical Descriptors
    print("Analyzing physicochemical descriptors...")
    physchem_feature_names = [
        "Molecular Weight", "LogP", "H-bond Donors", "H-bond Acceptors", 
        "TPSA", "Rotatable Bonds", "Rings", "Fraction sp3", 
        "Aromatic Rings", "Heteroatoms", "Molar Refractivity", 
        "Labute ASA", "QED"
    ]
    
    for feature_idx, feature_name in enumerate(physchem_feature_names):
        feature_preds = []
        
        # Fix: Use X_test consistently throughout the function
        feature_values = [X_test[i][2][0, feature_idx].item() for i in range(len(X_test))]
        feature_mean = np.mean(feature_values)
        
        with torch.no_grad():
            for idx, (features_adj_physchem_mol, _) in enumerate(zip(X_test, y_test)):
                features, adj, physchem, _ = features_adj_physchem_mol
                
                # Create modified copy with descriptor set to mean
                modified_physchem = physchem.clone()
                modified_physchem[0, feature_idx] = feature_mean
                
                _, pred = model(features, adj, modified_physchem)
                feature_preds.append(pred.item())
        
        feature_rmse = np.sqrt(mean_squared_error(y_test, feature_preds))
        importance_score = feature_rmse - baseline_rmse
        
        feature_data.append({
            'Feature': feature_name,
            'Category': 'Physicochemical',
            'Importance_Score': importance_score
        })
    
    # Create DataFrame and sort by importance
    df = pd.DataFrame(feature_data)
    df = df.sort_values('Importance_Score', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    if output_dir is None:
        output_dir = os.getcwd()
        
    csv_path = os.path.join(output_dir, 'feature_importance_complete.csv')
    df.to_csv(csv_path, index=False)
    print(f"Feature importance analysis saved to: {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    # Create colors by category
    colors = {
        'Graph-Atom Type': 'blue',
        'Graph-Atom Property': 'skyblue',
        'Graph-Structure': 'green', 
        'Physicochemical': 'orange'
    }
    
    # Plot top 20 features or all if less than 20
    top_n = min(20, len(df))
    top_df = df.head(top_n)
    
    bar_colors = [colors[category] for category in top_df['Category']]
    
    plt.barh(top_df['Feature'], top_df['Importance_Score'], color=bar_colors)
    
    # Add category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in colors.items()]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.xlabel('Increase in RMSE when feature is removed/modified')
    plt.title('Top Feature Importance Analysis')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_complete.png')
    plt.savefig(plot_path)
    plt.close()
    
    return df

# %% Main execution
def main():
    start_time = time.time()
    # Create model_output_evaluation directory and unique results folder inside it
    parent_output_dir = os.path.join(os.getcwd(), "model_output_evaluation")
    os.makedirs(parent_output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(parent_output_dir, f"results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting molecular solubility prediction at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output files will be saved in: {output_dir}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Step 1: Load ESOL dataset
    print("\n1. Loading ESOL dataset for molecular solubility prediction...")
    esol_df = load_esol_dataset()
    if esol_df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset shape: {esol_df.shape}")
    print("Sample data:")
    print(esol_df.head())
    
    # Step 2: Split into training and unseen prediction sets
    prediction_set_size = 0.1  # 10% of data kept as unseen prediction set
    print(f"\n2. Splitting dataset - keeping {prediction_set_size*100:.0f}% as unseen prediction set")
    
    train_df, prediction_df = train_test_split(esol_df, test_size=prediction_set_size, random_state=42)
    print(f"Training set: {len(train_df)} molecules")
    print(f"Unseen prediction set: {len(prediction_df)} molecules")
    
    # Step 3: Process data and train model
    print("\n3. Training model on ESOL dataset with physicochemical descriptors...")
    model, train_metrics, test_metrics, best_model_state = train_solubility_model(train_df, output_dir=output_dir)
    
    # Step 4: Evaluate performance
    print("\n4. Evaluating model performance")
    print(f"Training metrics: RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
    print(f"Test metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
    
    # Step 5: Keep best model
    print("\n5. Loading best model from training")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model loaded successfully")
    
    # Advanced model evaluation
    print("\n=== Advanced Model Evaluation ===")
    
    # Process molecules for evaluation
    converter = MoleculeGraphConverter()
    X_all = []
    y_all = []
    
    print("Processing molecules for evaluation...")
    for _, row in train_df.iterrows():
        try:
            smiles = row['smiles']
            solubility = row['solubility']
            
            mol_graph = converter.smiles_to_graph(smiles)
            if mol_graph:
                X_all.append((
                    mol_graph['node_features'], 
                    mol_graph['adj_matrix'], 
                    mol_graph['physchem_features'],
                    mol_graph['mol']
                ))
                y_all.append(solubility)
        except Exception as e:
            pass
    
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    # K-fold cross-validation - changed to 2-fold with 20 epochs each
    cv_results = ModelEvaluator.perform_kfold_cv(X_all, y_all, k=2, epochs=20)
    
    print("\nK-fold Cross-Validation Results:")
    print(f"Average RMSE: {cv_results['avg_metrics']['rmse']:.4f}")
    print(f"Average R²: {cv_results['avg_metrics']['r2']:.4f}")
    print(f"Average Pearson correlation: {cv_results['avg_metrics']['pearson_r']:.4f}")
    print(f"Average Spearman correlation: {cv_results['avg_metrics']['spearman_rho']:.4f}")
    
    # Feature importance analysis
    print("\nGenerating comprehensive feature importance analysis...")
    feature_imp_df = generate_feature_importance_csv(model, X_test, y_test, output_dir=output_dir)
    
    print("\nTop 5 most important features:")
    for idx, row in feature_imp_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']} ({row['Category']}): {row['Importance_Score']:.4f}")
    
    # Step 6: Predict solubility for unseen molecules
    print("\n6. Predicting solubility for unseen molecules")
    prediction_results = predict_unseen_molecules(model, prediction_df)
    
    # Step 7: Report results
    print("\n7. Final results")
    print(f"Unseen prediction set metrics:")
    print(f"  RMSE: {prediction_results['rmse']:.4f}")
    print(f"  MAE: {prediction_results['mae']:.4f}")
    print(f"  R²: {prediction_results['r2']:.4f}")
    
    # Save unseen predictions scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_results['actual'], prediction_results['predicted'], alpha=0.6)
    plt.plot([min(prediction_results['actual']), max(prediction_results['actual'])], 
             [min(prediction_results['actual']), max(prediction_results['actual'])], 'k--')
    plt.xlabel('Actual LogS')
    plt.ylabel('Predicted LogS')
    plt.title('Unseen Molecules: Predicted vs Actual Solubility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unseen_predictions.png'))
    plt.close()
    
    # Save all evaluation results to the summary file with comprehensive model description
    with open(os.path.join(output_dir, 'advanced_evaluation_results.txt'), 'w') as f:
        f.write(f"GNNSol Molecular Solubility Prediction Results\n")
        f.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add model architecture and feature information
        f.write(f"Model Architecture:\n")
        f.write(f"  Combined Graph Neural Network with physicochemical descriptors\n\n")
        
        f.write(f"Molecular Graph Features:\n")
        f.write(f"  - Atoms: H, C, N, O, F, P, S, Cl, Br, I (one-hot encoded)\n")
        f.write(f"  - Formal charge\n")
        f.write(f"  - Aromaticity\n")
        f.write(f"  - Atom degree (number of connections)\n")
        f.write(f"  - Number of attached hydrogens\n\n")
        
        f.write(f"Physicochemical Descriptors:\n")
        f.write(f"  - Molecular Weight\n")
        f.write(f"  - LogP (octanol-water partition coefficient)\n")
        f.write(f"  - Number of H-bond donors\n")
        f.write(f"  - Number of H-bond acceptors\n")
        f.write(f"  - Topological Polar Surface Area (TPSA)\n")
        f.write(f"  - Number of rotatable bonds\n")
        f.write(f"  - Number of rings\n")
        f.write(f"  - Fraction of sp3 carbons\n")
        f.write(f"  - Number of aromatic rings\n")
        f.write(f"  - Number of heteroatoms\n")
        f.write(f"  - Molar refractivity\n")
        f.write(f"  - Labute Accessible Surface Area\n")
        f.write(f"  - QED drug-likeness score\n\n")
        
        f.write(f"Training metrics:\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {train_metrics['mae']:.4f}\n")
        f.write(f"  R²: {train_metrics['r2']:.4f}\n\n")
        
        f.write(f"Test metrics:\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  R²: {test_metrics['r2']:.4f}\n\n")
        
        f.write(f"K-fold Cross-Validation Results (k=2):\n")
        f.write(f"  Average RMSE: {cv_results['avg_metrics']['rmse']:.4f}\n")
        f.write(f"  Average MAE: {cv_results['avg_metrics']['mae']:.4f}\n")
        f.write(f"  Average R²: {cv_results['avg_metrics']['r2']:.4f}\n")
        f.write(f"  Average Q²: {cv_results['avg_metrics']['q2']:.4f}\n")
        f.write(f"  Average Pearson correlation: {cv_results['avg_metrics']['pearson_r']:.4f}\n")
        f.write(f"  Average Spearman correlation: {cv_results['avg_metrics']['spearman_rho']:.4f}\n\n")
        
        f.write(f"Feature Importance (Top 5):\n")
        for i, row in enumerate(feature_imp_df.head(5).itertuples(index=False)):
            f.write(f"  {i+1}. {row.Feature} ({row.Category}): {row.Importance_Score:.4f}\n")
        f.write("\n")
        
        f.write(f"Unseen prediction set metrics:\n")
        f.write(f"  RMSE: {prediction_results['rmse']:.4f}\n")
        f.write(f"  MAE: {prediction_results['mae']:.4f}\n")
        f.write(f"  R²: {prediction_results['r2']:.4f}\n\n")
        
        f.write(f"Sample predictions:\n")
        for i in range(min(5, len(prediction_results['smiles']))):
            f.write(f"Molecule: {prediction_results['smiles'][i]}\n")
            f.write(f"  Actual: {prediction_results['actual'][i]:.4f}, Predicted: {prediction_results['predicted'][i]:.4f}, ")
            f.write(f"Error: {abs(prediction_results['actual'][i] - prediction_results['predicted'][i]):.4f}\n")
    
    # Save model state dictionary
    torch.save(best_model_state, os.path.join(output_dir, 'best_model.pt'))
    
    total_time = time.time() - start_time
    print(f"\nSolubility prediction completed in {total_time:.1f} seconds")
    print(f"Results saved in: {output_dir}")

def load_esol_dataset():
    """Load the ESOL dataset for molecular solubility prediction"""
    try:
        url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        df = pd.read_csv(url)
        print(f"Successfully loaded {len(df)} molecules from ESOL dataset")
        
        # Check for solubility column
        solubility_col = 'measured log solubility in mols per litre'
        if solubility_col not in df.columns:
            solubility_cols = [col for col in df.columns if 'solubility' in col.lower()]
            if solubility_cols:
                solubility_col = solubility_cols[0]
                print(f"Using solubility column: {solubility_col}")
            else:
                print("No solubility column found in dataset")
                return None
        
        # Keep only necessary columns
        df = df[['smiles', solubility_col, 'Compound ID']]
        df.rename(columns={solubility_col: 'solubility'}, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading ESOL dataset: {e}")
        return None

def train_solubility_model(train_df, epochs=300, batch_size=32, output_dir=None):
    """Train the GNN model on training data"""
    converter = MoleculeGraphConverter()
    solubility_col = 'solubility'  # We renamed it in load_esol_dataset
    
    # Process all molecules
    print("Processing molecules...")
    X = []
    y = []
    failed_count = 0
    
    for _, row in train_df.iterrows():
        try:
            smiles = row['smiles']
            solubility = row[solubility_col]
            
            mol_graph = converter.smiles_to_graph(smiles)
            if mol_graph:
                X.append((
                    mol_graph['node_features'], 
                    mol_graph['adj_matrix'], 
                    mol_graph['physchem_features'],
                    mol_graph['mol']
                ))
                y.append(solubility)
        except Exception as e:
            failed_count += 1
            if failed_count < 5:
                print(f"Error processing molecule {row['smiles']}: {e}")
            elif failed_count == 5:
                print("Additional errors occurred but not displayed...")
    
    print(f"Successfully processed {len(X)} molecules. Failed: {failed_count}")
    
    # Train-validation-test split (70-15-15)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)} molecules")
    
    # Create model
    feature_dim = X_train[0][0].shape[1]
    physchem_dim = X_train[0][2].shape[1]
    model = MolecularGNN(in_features=feature_dim, hidden_features=128, latent_dim=64, 
                         physchem_features=physchem_dim, out_features=1)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_model_state = None
    best_val_loss = float('inf')
    patience = 30
    no_improve = 0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Process in batches
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0
            
            for idx in batch_indices:
                features, adj, physchem, _ = X_train[idx]
                target = y_train[idx]
                
                optimizer.zero_grad()
                _, pred = model(features, adj, physchem)
                loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_loss += loss.item()
            
            train_loss += batch_loss / len(batch_indices)
        
        train_loss /= (len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0))
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for i, (features_adj_physchem_mol, target) in enumerate(zip(X_val, y_val)):
                features, adj, physchem, _ = features_adj_physchem_mol
                _, pred = model(features, adj, physchem)
                val_preds.append(pred.item())
                val_targets.append(target)
                
                loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                val_loss += loss.item()
        
        val_loss /= len(X_val)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs")
            break
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Final evaluation on training set
    model.eval()
    train_preds = []
    train_targets = []
    
    with torch.no_grad():
        for features_adj_physchem_mol, target in zip(X_train, y_train):
            features, adj, physchem, _ = features_adj_physchem_mol
            _, pred = model(features, adj, physchem)
            train_preds.append(pred.item())
            train_targets.append(target)
    
    train_metrics = {
        'rmse': np.sqrt(mean_squared_error(train_targets, train_preds)),
        'mae': mean_absolute_error(train_targets, train_preds),
        'r2': r2_score(train_targets, train_preds)
    }
    
    # Final evaluation on test set
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for features_adj_physchem_mol, target in zip(X_test, y_test):
            features, adj, physchem, _ = features_adj_physchem_mol
            _, pred = model(features, adj, physchem)
            test_preds.append(pred.item())
            test_targets.append(target)
    
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(test_targets, test_preds)),
        'mae': mean_absolute_error(test_targets, test_preds),
        'r2': r2_score(test_targets, test_preds)
    }
    
    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    else:
        plt.savefig('training_progress.png')
    plt.close()  # Close the figure instead of showing it
    
    return model, train_metrics, test_metrics, best_model_state

def predict_unseen_molecules(model, prediction_df):
    """Predict solubility for unseen molecules"""
    converter = MoleculeGraphConverter()
    solubility_col = 'solubility'
    
    predicted_values = []
    actual_values = []
    smiles_list = []
    failed_molecules = []
    
    print(f"Predicting solubility for {len(prediction_df)} unseen molecules...")
    
    for _, row in prediction_df.iterrows():
        try:
            smiles = row['smiles']
            actual_solubility = row[solubility_col]
            
            mol_graph = converter.smiles_to_graph(smiles)
            if mol_graph:
                # Make prediction
                model.eval()
                with torch.no_grad():
                    node_features = mol_graph['node_features']
                    adj_matrix = mol_graph['adj_matrix']
                    physchem_features = mol_graph['physchem_features']
                    _, pred = model(node_features, adj_matrix, physchem_features)
                    pred_value = pred.item()
                
                predicted_values.append(pred_value)
                actual_values.append(actual_solubility)
                smiles_list.append(smiles)
            else:
                failed_molecules.append((smiles, "Could not convert to graph"))
        except Exception as e:
            failed_molecules.append((smiles, str(e)))
    
    print(f"Successfully predicted {len(predicted_values)} molecules")
    print(f"Failed to predict {len(failed_molecules)} molecules")
    
    # Sample of predictions
    print("\nSample predictions:")
    for i in range(min(5, len(predicted_values))):
        print(f"Molecule: {smiles_list[i]}")
        print(f"  Actual: {actual_values[i]:.4f}, Predicted: {predicted_values[i]:.4f}, "
              f"Error: {abs(actual_values[i] - predicted_values[i]):.4f}")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mae = mean_absolute_error(actual_values, predicted_values)
    r2 = r2_score(actual_values, predicted_values)
    
    # Return results
    return {
        'predicted': predicted_values,
        'actual': actual_values,
        'smiles': smiles_list,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'failed': failed_molecules
    }

if __name__ == "__main__":
    main()
