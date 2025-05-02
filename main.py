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
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time
import datetime
import os

# %% Molecular Graph Construction
class MoleculeGraphConverter:
    def __init__(self):
        # Atom features: Atomic number, hybridization, aromaticity, num H, formal charge
        self.atom_features = {
            'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35, 53],  # H, C, N, O, F, P, S, Cl, Br, I
        }
        
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
        
        return {
            'mol': mol,
            'adj_matrix': adj_matrix,
            'node_features': torch.tensor(node_features, dtype=torch.float),
            'smiles': smiles
        }

# %% GNN Model Definition
class MolecularGNN(torch.nn.Module):
    def __init__(self, in_features, hidden_features, latent_dim, out_features):
        super(MolecularGNN, self).__init__()
        # Encoding layers
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.fc2 = torch.nn.Linear(hidden_features, hidden_features)
        self.fc3 = torch.nn.Linear(hidden_features, latent_dim)
        
        # Property prediction head
        self.fc_out = torch.nn.Linear(latent_dim, out_features)
        
        # Activation functions
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x, adj_matrix):
        # Message passing layers (graph convolution)
        x = self.fc1(x)
        x = self.relu(torch.mm(adj_matrix, x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(torch.mm(adj_matrix, x))
        x = self.dropout(x)
        
        # Encoding to latent space
        latent = self.fc3(x)
        
        # Global pooling (mean of node embeddings)
        graph_embedding = torch.mean(latent, dim=0, keepdim=True)
        
        # Property prediction
        out = self.fc_out(graph_embedding)
        return latent, out

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
    print("\n3. Training model on ESOL dataset...")
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
    
    # Step 6: Predict solubility for unseen molecules
    print("\n6. Predicting solubility for unseen molecules")
    prediction_results = predict_unseen_molecules(model, prediction_df)
    
    # Step 7: Report results
    print("\n7. Final results")
    print(f"Unseen prediction set metrics:")
    print(f"  RMSE: {prediction_results['rmse']:.4f}")
    print(f"  MAE: {prediction_results['mae']:.4f}")
    print(f"  R²: {prediction_results['r2']:.4f}")
    
    # Save prediction scatter plot
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
    plt.close()  # Close the figure instead of showing it
    
    # Save residuals plot
    plt.figure(figsize=(10, 6))
    residuals = np.array(prediction_results['predicted']) - np.array(prediction_results['actual'])
    plt.scatter(prediction_results['predicted'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted LogS')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()  # Close the figure instead of showing it
    
    # Save a summary of results to a text file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"GNNSol Molecular Solubility Prediction Results\n")
        f.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Training metrics:\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {train_metrics['mae']:.4f}\n")
        f.write(f"  R²: {train_metrics['r2']:.4f}\n\n")
        f.write(f"Test metrics:\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  R²: {test_metrics['r2']:.4f}\n\n")
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
                X.append((mol_graph['node_features'], mol_graph['adj_matrix'], mol_graph['mol']))
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
    model = MolecularGNN(in_features=feature_dim, hidden_features=128, latent_dim=64, out_features=1)
    
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
                features, adj, _ = X_train[idx]
                target = y_train[idx]
                
                optimizer.zero_grad()
                _, pred = model(features, adj)
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
            for i, (features_adj_mol, target) in enumerate(zip(X_val, y_val)):
                features, adj, _ = features_adj_mol
                _, pred = model(features, adj)
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
        for features_adj_mol, target in zip(X_train, y_train):
            features, adj, _ = features_adj_mol
            _, pred = model(features, adj)
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
        for features_adj_mol, target in zip(X_test, y_test):
            features, adj, _ = features_adj_mol
            _, pred = model(features, adj)
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
                    _, pred = model(node_features, adj_matrix)
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
