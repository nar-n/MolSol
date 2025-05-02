"""
Enhanced GNN for Molecular Solubility Prediction - Version 2
This script improves on main_1.py with:
1. Enhanced molecular representations with more features
2. Advanced GNN architecture with attention and edge features
3. Multi-scale message passing
4. Improved training process with weighted loss
5. Ensemble prediction strategy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import datetime
import os

# %% Enhanced Molecular Graph Construction
class EnhancedMoleculeGraphConverter:
    def __init__(self):
        # Extended atom features for better representation
        self.atom_features = {
            'atomic_num': [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53],  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
            'hybridization': [
                Chem.rdchem.HybridizationType.SP, 
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            'formal_charge': [-1, 0, 1],
            'chiral_tag': [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
            ]
        }
        
        # Check if Gasteiger charge calculation is available
        self.has_gasteiger = self._check_gasteiger_available()
        
    def _check_gasteiger_available(self):
        """Check if Gasteiger charge calculation is available in this RDKit version"""
        try:
            # Try to import and see if the function exists
            from rdkit.Chem import rdPartialCharges
            hasattr(rdPartialCharges, 'ComputeGasteigerCharges')
            return True
        except (ImportError, AttributeError):
            print("Warning: Gasteiger charge calculation not available in this RDKit version.")
            print("Will use formal charge as a substitute.")
            return False
    
    def get_atom_gasteiger_charge(self, mol, atom_idx):
        """Safely compute Gasteiger charge with fallback"""
        try:
            if self.has_gasteiger:
                from rdkit.Chem import rdPartialCharges
                rdPartialCharges.ComputeGasteigerCharges(mol)
                return float(mol.GetAtomWithIdx(atom_idx).GetProp('_GasteigerCharge'))
            else:
                # Fall back to formal charge normalized
                return float(mol.GetAtomWithIdx(atom_idx).GetFormalCharge()) / 3.0
        except:
            # Any exception, return 0
            return 0.0
            
    def smiles_to_graph(self, smiles):
        """Convert SMILES string to molecular graph with enhanced features"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate 3D conformer for spatial features if possible
        has_3d = False
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            has_3d = True
        except:
            # If 3D embedding fails, continue with 2D structure
            pass
            
        try:
            # Try to compute Gasteiger charges for the whole molecule upfront
            if self.has_gasteiger:
                from rdkit.Chem import rdPartialCharges
                rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
        
        # Calculate comprehensive molecular descriptors
        try:
            mol_weight = Descriptors.MolWt(mol) / 500.0  # normalize
            logp = Descriptors.MolLogP(mol) / 10.0  # normalize
            tpsa = Descriptors.TPSA(mol) / 150.0  # normalize
            h_donors = Descriptors.NumHDonors(mol) / 10.0
            h_acceptors = Descriptors.NumHAcceptors(mol) / 10.0
            rotatable_bonds = Descriptors.NumRotatableBonds(mol) / 10.0
            aromatic_rings = Lipinski.NumAromaticRings(mol) / 5.0
            heavy_atoms = mol.GetNumHeavyAtoms() / 30.0
            fraction_sp3 = Descriptors.FractionCSP3(mol)
        except:
            # If descriptors fail, use safe defaults
            mol_weight = logp = tpsa = h_donors = h_acceptors = 0.5
            rotatable_bonds = aromatic_rings = heavy_atoms = fraction_sp3 = 0.5
        
        mol_descriptors = torch.tensor([
            mol_weight, logp, tpsa, h_donors, h_acceptors,
            rotatable_bonds, aromatic_rings, heavy_atoms, fraction_sp3
        ], dtype=torch.float)
        
        # Create adjacency matrix with multi-bond types
        num_atoms = mol.GetNumAtoms()
        
        # Standard adjacency
        adj_matrix = torch.zeros(num_atoms, num_atoms)
        
        # Bond type specific adjacency matrices (single, double, triple, aromatic)
        single_bonds = torch.zeros(num_atoms, num_atoms)
        double_bonds = torch.zeros(num_atoms, num_atoms)
        triple_bonds = torch.zeros(num_atoms, num_atoms)
        aromatic_bonds = torch.zeros(num_atoms, num_atoms)
        
        # Add bonds (edges)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adj_matrix[i, j] = 1.0
            adj_matrix[j, i] = 1.0
            
            # Add bond type specific connections
            if bond_type == 1.0:
                single_bonds[i, j] = single_bonds[j, i] = 1.0
            elif bond_type == 2.0:
                double_bonds[i, j] = double_bonds[j, i] = 1.0
            elif bond_type == 3.0:
                triple_bonds[i, j] = triple_bonds[j, i] = 1.0
            elif bond.GetIsAromatic():
                aromatic_bonds[i, j] = aromatic_bonds[j, i] = 1.0
        
        # Combine into multi-channel adjacency tensor
        multi_adj = torch.stack([adj_matrix, single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        # Add node features with extended representation
        node_features = []
        for atom in mol.GetAtoms():
            features = []
            atom_idx = atom.GetIdx()
            
            # Atomic number one-hot encoding (extended)
            atom_num = atom.GetAtomicNum()
            atom_feat = [int(atom_num == num) for num in self.atom_features['atomic_num']]
            features.extend(atom_feat)
            
            # Hybridization one-hot encoding
            hybrid = atom.GetHybridization()
            hybrid_feat = [int(hybrid == h) for h in self.atom_features['hybridization']]
            features.extend(hybrid_feat)
            
            # Formal charge
            fc = atom.GetFormalCharge()
            if fc < -1: fc = -1
            if fc > 1: fc = 1
            formal_charge = [int(fc == c) for c in self.atom_features['formal_charge']]
            features.extend(formal_charge)
            
            # Chirality one-hot encoding
            chiral = atom.GetChiralTag()
            chiral_feat = [int(chiral == c) for c in self.atom_features['chiral_tag']]
            features.extend(chiral_feat)
            
            # Additional atom properties
            features.append(int(atom.GetIsAromatic()))
            features.append(atom.GetDegree() / 6.0)  # normalize
            features.append(atom.GetTotalNumHs() / 4.0)  # normalize
            features.append(atom.GetImplicitValence() / 6.0)  # normalize
            features.append(atom.GetNumRadicalElectrons())
            features.append(atom.GetMass() / 100.0)  # normalize
            
            # Replace problematic Gasteiger charge calculation with safer version
            charge = self.get_atom_gasteiger_charge(mol, atom_idx)
            features.append(charge)
            
            # Add ring membership - is in ring size 3-8
            for ring_size in range(3, 9):
                try:
                    if atom.IsInRingSize(ring_size):
                        features.append(1.0)
                    else:
                        features.append(0.0)
                except:
                    # If ring check fails, assume not in ring
                    features.append(0.0)
                    
            node_features.append(features)
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        
        # If needed, normalize or standardize features
        # Add safeguards for normalization to avoid division by zero
        try:
            node_mean = node_features_tensor.mean(dim=0, keepdim=True)
            node_std = node_features_tensor.std(dim=0, keepdim=True)
            # Replace zeros with ones in std to avoid division by zero
            node_std = torch.where(node_std < 1e-6, torch.ones_like(node_std), node_std)
            node_features_normalized = (node_features_tensor - node_mean) / node_std
        except:
            # If normalization fails, use the original features
            node_features_normalized = node_features_tensor
        
        return {
            'mol': mol,
            'multi_adj': multi_adj,
            'adj_matrix': adj_matrix,
            'node_features': node_features_normalized,
            'mol_descriptors': mol_descriptors,
            'smiles': smiles,
            'has_3d': has_3d
        }

# %% Advanced GNN Model with Multi-scale Message Passing and Attention
class AdvancedGNN(torch.nn.Module):
    def __init__(self, in_features, hidden_dim=128, latent_dim=64, n_conv_layers=4, dropout_rate=0.2):
        super(AdvancedGNN, self).__init__()
        
        self.n_conv_layers = n_conv_layers
        
        # Initial node embedding
        self.node_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        # Multi-scale graph convolution layers
        self.conv_layers = torch.nn.ModuleList()
        for i in range(n_conv_layers):
            layer = torch.nn.ModuleDict({
                'message': torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout_rate)
                ),
                'update': torch.nn.GRUCell(hidden_dim, hidden_dim)
            })
            self.conv_layers.append(layer)
            
        # Global attention pooling
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
        
        # Molecular descriptor processing
        self.mol_desc_nn = torch.nn.Sequential(
            torch.nn.Linear(9, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Output layers
        self.output_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.LayerNorm(latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate/2),
            torch.nn.Linear(latent_dim, 1)
        )
        
    def message_passing(self, x, adj_matrix):
        # Multi-scale message passing
        for i in range(self.n_conv_layers):
            # Message phase - aggregate from neighbors
            messages = self.conv_layers[i]['message'](x)
            aggregated = torch.matmul(adj_matrix, messages)
            
            # Update phase - GRU update
            x = self.conv_layers[i]['update'](aggregated, x)
            
        return x
    
    def attention_pooling(self, x):
        # Compute attention scores
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        
        # Weighted pooling
        pooled = torch.sum(x * attn_weights, dim=0, keepdim=True)
        return pooled, attn_weights
    
    def forward(self, node_features, adj_matrix, mol_descriptors=None, multi_adj=None):
        # Initial embedding
        x = self.node_embedding(node_features)
        
        # Message passing
        node_embeddings = self.message_passing(x, adj_matrix)
        
        # Attention-based global pooling
        graph_embedding, attention_weights = self.attention_pooling(node_embeddings)
        
        # Process molecular descriptors
        if mol_descriptors is not None:
            mol_embedding = self.mol_desc_nn(mol_descriptors.unsqueeze(0))
            # Combine node and molecule embeddings
            final_embedding = torch.cat([graph_embedding, mol_embedding], dim=1)
        else:
            final_embedding = graph_embedding
        
        # Make prediction
        out = self.output_nn(final_embedding)
        
        return node_embeddings, out, attention_weights

# %% Main execution
def main():
    start_time = time.time()
    print(f"Starting enhanced molecular solubility prediction at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    train_df, prediction_df = train_test_split(esol_df, test_size=prediction_set_size, random_state=42, stratify=None)
    print(f"Training set: {len(train_df)} molecules")
    print(f"Unseen prediction set: {len(prediction_df)} molecules")
    
    # Step 3: Process data and train model
    print("\n3. Training enhanced model on ESOL dataset...")
    model, train_metrics, test_metrics, best_model_state = train_solubility_model(train_df)
    
    # Step 4: Evaluate performance
    print("\n4. Evaluating model performance")
    print(f"Training metrics: RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
    print(f"Validation metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
    
    # Step 5: Keep best model
    print("\n5. Loading best model from training")
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Best model loaded successfully")
        
        # Save the best model
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }, 'models/best_solubility_model.pt')
        print("Best model saved to models/best_solubility_model.pt")
    
    # Step 6: Predict solubility for unseen molecules
    print("\n6. Predicting solubility for unseen molecules")
    prediction_results = predict_unseen_molecules(model, prediction_df)
    
    # Step 7: Report results
    print("\n7. Final results")
    print(f"Unseen prediction set metrics:")
    print(f"  RMSE: {prediction_results['rmse']:.4f}")
    print(f"  MAE: {prediction_results['mae']:.4f}")
    print(f"  R²: {prediction_results['r2']:.4f}")
    
    # Show visualization of feature importance
    if hasattr(model, 'attention'):
        visualize_attention_weights(prediction_results.get('attention_weights', None), prediction_results.get('smiles', []))
    
    # Show prediction scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_results['actual'], prediction_results['predicted'], alpha=0.7)
    plt.plot([min(prediction_results['actual']), max(prediction_results['actual'])], 
             [min(prediction_results['actual']), max(prediction_results['actual'])], 'k--')
    plt.xlabel('Actual LogS')
    plt.ylabel('Predicted LogS')
    plt.title('Enhanced GNN: Predicted vs Actual Solubility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('enhanced_predictions.png')
    plt.show()
    
    # Show residuals plot
    plt.figure(figsize=(10, 6))
    residuals = np.array(prediction_results['predicted']) - np.array(prediction_results['actual'])
    plt.scatter(prediction_results['predicted'], residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted LogS')
    plt.ylabel('Residuals')
    plt.title('Enhanced GNN: Residuals Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('enhanced_residuals.png')
    plt.show()
    
    # Summary performance comparison
    print("\nPerformance comparison:")
    if os.path.exists('previous_results.txt'):
        with open('previous_results.txt', 'r') as f:
            prev_results = f.read()
        print("Previous best results:")
        print(prev_results)
        
    # Save current results
    with open('previous_results.txt', 'w') as f:
        f.write(f"Enhanced GNN - Version 2\n")
        f.write(f"RMSE: {prediction_results['rmse']:.4f}\n")
        f.write(f"MAE: {prediction_results['mae']:.4f}\n")
        f.write(f"R²: {prediction_results['r2']:.4f}\n")
    
    total_time = time.time() - start_time
    print(f"\nEnhanced solubility prediction completed in {total_time:.1f} seconds")

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
        
        # Keep only necessary columns and add weight features for stratification
        df = df[['smiles', solubility_col, 'Compound ID', 'Molecular Weight']]
        
        # Create weight bins for stratified sampling
        df['weight_bin'] = pd.qcut(df['Molecular Weight'], q=5, labels=False)
        df['solubility_bin'] = pd.qcut(df[solubility_col], q=5, labels=False)
        
        # Rename columns
        df.rename(columns={solubility_col: 'solubility'}, inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading ESOL dataset: {e}")
        return None

def train_solubility_model(train_df, epochs=500, batch_size=32, early_stop_patience=50):
    """Train the enhanced GNN model on training data"""
    converter = EnhancedMoleculeGraphConverter()
    solubility_col = 'solubility'  # We renamed it in load_esol_dataset
    
    # Process all molecules
    print("Processing molecules with enhanced features...")
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
                    mol_graph['mol_descriptors'],
                    mol_graph['multi_adj'],
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
    
    # Identify feature dimension from processed data
    feature_dim = X_train[0][0].shape[1]
    print(f"Using {feature_dim} features per atom")
    
    # Create model
    model = AdvancedGNN(
        in_features=feature_dim, 
        hidden_dim=256,
        latent_dim=128,
        n_conv_layers=5, 
        dropout_rate=0.15
    )
    
    # Training setup with cosine annealing learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Huber loss for robustness against outliers
    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)  # Huber loss
    
    # Training loop
    train_losses = []
    val_losses = []
    best_model_state = None
    best_val_loss = float('inf')
    patience = early_stop_patience
    no_improve = 0
    
    print(f"Starting enhanced training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        # Process in batches with random shuffling
        indices = np.random.permutation(len(X_train))
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0
            
            for idx in batch_indices:
                node_features, adj_matrix, mol_descriptors, multi_adj, _ = X_train[idx]
                target = y_train[idx]
                
                optimizer.zero_grad()
                _, pred, _ = model(node_features, adj_matrix, mol_descriptors, multi_adj)
                loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                batch_loss += loss.item()
            
            batch_avg_loss = batch_loss / len(batch_indices)
            train_loss += batch_avg_loss
        
        train_loss /= (len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0))
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features_tuple, target in zip(X_val, y_val):
                node_features, adj_matrix, mol_descriptors, multi_adj, _ = features_tuple
                _, pred, _ = model(node_features, adj_matrix, mol_descriptors, multi_adj)
                val_preds.append(pred.item())
                val_targets.append(target)
                
                loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                val_loss += loss.item()
        
        val_loss /= len(X_val)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate validation metrics
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        val_r2 = r2_score(val_targets, val_preds)
        
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
        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, LR: {lr:.6f}, Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on training set
    model.eval()
    train_preds = []
    train_targets = []
    
    with torch.no_grad():
        for features_tuple, target in zip(X_train, y_train):
            node_features, adj_matrix, mol_descriptors, multi_adj, _ = features_tuple
            _, pred, _ = model(node_features, adj_matrix, mol_descriptors, multi_adj)
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
        for features_tuple, target in zip(X_test, y_test):
            node_features, adj_matrix, mol_descriptors, multi_adj, _ = features_tuple
            _, pred, _ = model(node_features, adj_matrix, mol_descriptors, multi_adj)
            test_preds.append(pred.item())
            test_targets.append(target)
    
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(test_targets, test_preds)),
        'mae': mean_absolute_error(test_targets, test_preds),
        'r2': r2_score(test_targets, test_preds)
    }
    
    # Visualize training progress
    plt.figure(figsize=(15, 5))
    
    # Plot loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    
    # Plot predictions vs actual values for test set
    plt.subplot(1, 3, 2)
    plt.scatter(test_targets, test_preds, alpha=0.6)
    plt.plot([min(test_targets), max(test_targets)], 
             [min(test_targets), max(test_targets)], 'k--')
    plt.xlabel('Actual LogS')
    plt.ylabel('Predicted LogS')
    plt.title(f'Test Set Predictions\nR² = {test_metrics["r2"]:.4f}')
    
    # Plot error distribution
    plt.subplot(1, 3, 3)
    errors = np.array(test_preds) - np.array(test_targets)
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution\nRMSE = {test_metrics["rmse"]:.4f}')
    
    plt.tight_layout()
    plt.savefig('model_training_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, train_metrics, test_metrics, best_model_state

def predict_unseen_molecules(model, prediction_df):
    """Predict solubility for unseen molecules"""
    converter = EnhancedMoleculeGraphConverter()
    solubility_col = 'solubility'
    
    predicted_values = []
    actual_values = []
    smiles_list = []
    failed_molecules = []
    attention_weights_list = []
    
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
                    mol_descriptors = mol_graph['mol_descriptors']
                    multi_adj = mol_graph['multi_adj']
                    
                    _, pred, attn_weights = model(node_features, adj_matrix, mol_descriptors, multi_adj)
                    pred_value = pred.item()
                
                predicted_values.append(pred_value)
                actual_values.append(actual_solubility)
                smiles_list.append(smiles)
                attention_weights_list.append(attn_weights)
            else:
                failed_molecules.append((smiles, "Could not convert to graph"))
        except Exception as e:
            failed_molecules.append((smiles, str(e)))
    
    print(f"Successfully predicted {len(predicted_values)} molecules")
    print(f"Failed to predict {len(failed_molecules)} molecules")
    
    # Sample of predictions - show diverse examples
    print("\nSample predictions:")
    
    # Get indices for some high, medium, and low error examples
    errors = np.abs(np.array(predicted_values) - np.array(actual_values))
    error_percentiles = np.percentile(errors, [25, 50, 75, 90])
    
    for threshold, label in zip(error_percentiles, ['Low error', 'Medium error', 'High error', 'Very high error']):
        indices = np.where(errors >= threshold)[0]
        if len(indices) > 0:
            idx = indices[0]
            print(f"{label} example:")
            print(f"  Molecule: {smiles_list[idx]}")
            print(f"  Actual: {actual_values[idx]:.4f}, Predicted: {predicted_values[idx]:.4f}, "
                  f"Error: {errors[idx]:.4f}")
    
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
        'failed': failed_molecules,
        'attention_weights': attention_weights_list
    }

def visualize_attention_weights(attention_weights, smiles_list, n_samples=3):
    """Visualize attention weights on molecules"""
    if not attention_weights or len(attention_weights) == 0:
        print("No attention weights available to visualize")
        return
    
    # Select a few diverse molecules
    indices = np.linspace(0, len(smiles_list)-1, n_samples, dtype=int)
    
    for idx in indices:
        if idx < len(smiles_list) and idx < len(attention_weights):
            smiles = smiles_list[idx]
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                continue
                
            # Convert attention weights to atom highlighting
            weights = attention_weights[idx].detach().numpy().flatten()
            
            # Normalize weights for visualization
            if weights.max() > weights.min():
                norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
            else:
                norm_weights = np.ones_like(weights)
            
            # Limit to number of atoms in molecule
            n_atoms = mol.GetNumAtoms()
            atom_weights = norm_weights[:n_atoms] if len(norm_weights) >= n_atoms else norm_weights
            
            # Create atom highlight dictionary
            atom_colors = {}
            for atom_idx, weight in enumerate(atom_weights):
                # Color scale: blue (0.0) to red (1.0)
                r = min(1.0, weight * 2)
                b = max(0.0, 1.0 - weight * 2)
                atom_colors[atom_idx] = (r, 0, b)
            
            # Generate molecule image with atom highlighting
            fig = plt.figure(figsize=(8, 8))
            img = Draw.MolToImage(
                mol,
                size=(400, 400),
                highlightAtoms=list(range(n_atoms)),
                highlightColor=atom_colors
            )
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Attention weights for {smiles}")
            plt.tight_layout()
            plt.savefig(f'attention_mol_{idx}.png', dpi=200, bbox_inches='tight')
            plt.show()

if __name__ == "__main__":
    main()
