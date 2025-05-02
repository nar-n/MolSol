# %% Imports
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import Draw
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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

# %% GNN Model Definition for Molecular Property Prediction
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

# %% Sample Data
# Sample SMILES with solubility data (LogS values)
sample_data = [
    ("CC(=O)OC1=CC=CC=C1C(=O)O", -1.72),  # Aspirin
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", -0.79),  # Caffeine
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", -3.97),  # Ibuprofen
    ("CN1C=NC2=C1C(=O)NC(=O)N2C", -1.16),  # Theophylline
    ("CCO", 0.14),  # Ethanol
    ("C1=CC=C(C=C1)O", -0.05),  # Phenol
    ("CC(C)(C)C", -3.21),  # tert-Butane
    ("CCCCCC", -3.76),  # Hexane
]

# %% Setup and Run Model
if __name__ == "__main__":
    print("Molecular GNN application starting...")
    print(f"PyTorch version: {torch.__version__}")

    # Initialize SMILES to graph converter
    converter = MoleculeGraphConverter()

    # %% Process a sample molecule
    smiles = sample_data[0][0]  # Aspirin
    mol_graph = converter.smiles_to_graph(smiles)

    if mol_graph:
        adj_matrix = mol_graph['adj_matrix']
        node_features = mol_graph['node_features']
        mol = mol_graph['mol']
        
        # Display molecule
        print(f"Processing molecule: {smiles}")
        print(f"Number of atoms: {mol.GetNumAtoms()}")
        print(f"Node features shape: {node_features.shape}")
        
        # Create a molecular GNN model
        feature_dim = node_features.shape[1]
        model = MolecularGNN(in_features=feature_dim, hidden_features=64, latent_dim=32, out_features=1)
        
        # Forward pass
        latent, output = model(node_features, adj_matrix)
        
        print(f"Latent representation shape: {latent.shape}")
        print(f"Predicted property: {output.item():.4f}")
        print(f"Actual LogS value: {sample_data[0][1]}")
        
        # Visualize the molecule
        img = Draw.MolToImage(mol, size=(400, 400))
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{Chem.MolToSmiles(mol)}\nActual LogS: {sample_data[0][1]}")
        plt.show()

    # %% Process and visualize all molecules - Training function
    def train_model(model, data, epochs=100):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        # Process all molecules
        X = []
        y = []
        
        for smiles, logs in data:
            mol_graph = converter.smiles_to_graph(smiles)
            if mol_graph:
                X.append((mol_graph['node_features'], mol_graph['adj_matrix'], mol_graph['mol']))
                y.append(logs)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            # Train on each molecule
            for (features, adj, _), target in zip(X_train, y_train):
                optimizer.zero_grad()
                _, pred = model(features, adj)
                loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            
            # Evaluate on test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for (features, adj, _), target in zip(X_test, y_test):
                    _, pred = model(features, adj)
                    loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                    test_loss += loss.item()
            
            test_loss /= len(X_test)
            test_losses.append(test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        return model, train_losses, test_losses, X_test, y_test

    # Train the model
    print("\nTraining model on sample data...")
    feature_dim = node_features.shape[1]
    full_model = MolecularGNN(in_features=feature_dim, hidden_features=64, latent_dim=32, out_features=1)
    trained_model, train_losses, test_losses, X_test, y_test = train_model(full_model, sample_data, epochs=50)

    # %% Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.show()
    
    # %% Evaluate model predictions
    print("\nEvaluating model predictions...")
    trained_model.eval()
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for (features, adj, mol), target in zip(X_test, y_test):
            _, pred = trained_model(features, adj)
            pred_value = pred.item()
            predictions.append(pred_value)
            actual_values.append(target)
            print(f"Molecule: {Chem.MolToSmiles(mol)}")
            print(f"  Actual: {target:.4f}, Predicted: {pred_value:.4f}, Error: {abs(target - pred_value):.4f}")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    r2 = r2_score(actual_values, predictions)
    print(f"\nOverall metrics - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_values, predictions)
    plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'k--')
    plt.xlabel('Actual LogS')
    plt.ylabel('Predicted LogS')
    plt.title('Model Predictions vs Actual Values')
    plt.grid(True)
    plt.show()
    
    # %% Load and train on large solubility dataset (ESOL)
    print("\n" + "="*50)
    print("TRAINING ON LARGE SOLUBILITY DATASET (ESOL)")
    print("="*50)
    
    def load_esol_dataset():
        """Load the ESOL dataset for molecular solubility prediction"""
        print("Loading ESOL dataset for molecular solubility prediction...")
        
        try:
            # Try to download the ESOL dataset
            url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
            df = pd.read_csv(url)
            print(f"Successfully loaded {len(df)} molecules from ESOL dataset")
            print("Available columns:", df.columns.tolist())
        except Exception as e:
            print(f"Error loading ESOL dataset: {e}")
            print("Using backup simplified dataset...")
            # If download fails, use a minimal dataset as backup
            df = pd.DataFrame({
                'smiles': [item[0] for item in sample_data],
                'measured log solubility in mols per litre': [item[1] for item in sample_data],
                'Compound ID': ['Compound_' + str(i) for i in range(len(sample_data))]
            })
        
        return df
    
    # Load the ESOL dataset
    esol_df = load_esol_dataset()
    print(f"Dataset shape: {esol_df.shape}")
    print(esol_df.head())
    
    def train_on_esol(max_molecules=800, epochs=300):
        """Train a GNN model on the ESOL dataset"""
        if 'measured log solubility in mols per litre' not in esol_df.columns:
            print("Could not find solubility column. Using ESOL predicted values instead.")
            solubility_col = [col for col in esol_df.columns if 'solubility' in col.lower()][0]
        else:
            solubility_col = 'measured log solubility in mols per litre'
            
        print(f"Using solubility values from column: {solubility_col}")
        
        # Sample a subset if dataset is large
        if len(esol_df) > max_molecules:
            df_subset = esol_df.sample(max_molecules, random_state=42)
        else:
            df_subset = esol_df
        
        print(f"Training on {len(df_subset)} molecules from ESOL dataset")
        
        # Process all molecules
        X = []
        y = []
        failed_count = 0
        
        for _, row in df_subset.iterrows():
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
        
        # Train-test split with 15% test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        
        # Create a new model
        if len(X_train) > 0:
            feature_dim = X_train[0][0].shape[1]
            model = MolecularGNN(in_features=feature_dim, hidden_features=128, latent_dim=64, out_features=1)
            
            # Training with AdamW optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
            loss_fn = torch.nn.MSELoss()
            
            # Training loop
            train_losses = []
            test_losses = []
            best_model = None
            best_rmse = float('inf')
            
            print(f"Starting training for {epochs} epochs...")
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                
                # Train on each molecule
                for (features, adj, _), target in zip(X_train, y_train):
                    optimizer.zero_grad()
                    _, pred = model(features, adj)
                    loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(X_train)
                train_losses.append(train_loss)
                
                # Evaluate on test set
                model.eval()
                test_loss = 0
                test_preds = []
                test_targets = []
                
                with torch.no_grad():
                    for (features, adj, _), target in zip(X_test, y_test):
                        _, pred = model(features, adj)
                        pred_value = pred.item()
                        test_preds.append(pred_value)
                        test_targets.append(target)
                        
                        loss = loss_fn(pred, torch.tensor([[target]], dtype=torch.float))
                        test_loss += loss.item()
                
                test_loss /= len(X_test)
                test_losses.append(test_loss)
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
                r2 = r2_score(test_targets, test_preds)
                
                scheduler.step(rmse)
                
                # Save best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model.state_dict().copy()
                
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Load best model
            if best_model:
                model.load_state_dict(best_model)
            
            print(f"\nTraining complete! Best RMSE: {best_rmse:.4f}")
            
            # Final evaluation
            model.eval()
            test_preds = []
            test_targets = []
            
            with torch.no_grad():
                for (features, adj, _), target in zip(X_test, y_test):
                    _, pred = model(features, adj)
                    test_preds.append(pred.item())
                    test_targets.append(target)
            
            # Calculate final metrics
            rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
            r2 = r2_score(test_targets, test_preds)
            
            print(f"Final Test RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # Plot predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(test_targets, test_preds, alpha=0.6)
            plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'k--')
            plt.xlabel('Actual LogS')
            plt.ylabel('Predicted LogS')
            plt.title('ESOL Model: Predicted vs Actual Solubility')
            plt.grid(True)
            plt.show()
            
            return model, X_test, y_test
        else:
            print("No valid molecules were processed. Cannot train the model.")
            return None, None, None
    
    # Train on the ESOL dataset
    esol_model, esol_X_test, esol_y_test = train_on_esol(max_molecules=800, epochs=300)
    
    # %% Predict solubility for unseen molecules
    def predict_solubility(model, smiles_list):
        """Predict solubility for a list of unseen molecules"""
        print("\n" + "="*50)
        print("PREDICTING SOLUBILITY FOR UNSEEN MOLECULES")
        print("="*50)
        
        results = []
        
        for smiles in smiles_list:
            try:
                # Process the molecule
                mol_graph = converter.smiles_to_graph(smiles)
                if mol_graph:
                    node_features = mol_graph['node_features']
                    adj_matrix = mol_graph['adj_matrix']
                    mol = mol_graph['mol']
                    
                    # Make prediction
                    model.eval()
                    with torch.no_grad():
                        _, pred = model(node_features, adj_matrix)
                        pred_value = pred.item()
                    
                    # Get molecule name or formula
                    mol_name = Chem.MolToSmiles(mol)
                    
                    print(f"Molecule: {mol_name}")
                    print(f"Predicted solubility (LogS): {pred_value:.4f}")
                    
                    # Visualize the molecule
                    img = Draw.MolToImage(mol, size=(300, 300))
                    plt.figure(figsize=(5, 5))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"{mol_name}\nPredicted LogS: {pred_value:.4f}")
                    plt.show()
                    
                    results.append((mol, mol_name, pred_value))
                else:
                    print(f"Failed to process SMILES: {smiles}")
            except Exception as e:
                print(f"Error predicting for {smiles}: {e}")
        
        return results
    
    # Test the trained model on unseen molecules
    if esol_model:
        # Example unseen molecules
        unseen_molecules = [
            "C1=CC=CC(=C1)C(=O)O",  # Benzoic acid
            "CC(C)COC(=O)C=C",       # Isobutyl acrylate
            "CCS(=O)(=O)C1=CC=CC=C1", # Ethyl benzenesulfonate
            "CC(C)(C)C(=O)OC1=CC=CC=C1", # Phenyl pivalate
            "CC(=O)OCCOC(=O)C"        # Ethylene glycol diacetate
        ]
        
        prediction_results = predict_solubility(esol_model, unseen_molecules)
        
        # Compare different molecules in a bar chart
        if len(prediction_results) > 1:
            plt.figure(figsize=(12, 6))
            molecules = [result[1][:20] + '...' if len(result[1]) > 20 else result[1] for result in prediction_results]
            values = [result[2] for result in prediction_results]
            
            plt.bar(molecules, values)
            plt.xlabel('Molecule')
            plt.ylabel('Predicted LogS')
            plt.title('Predicted Solubility for Unseen Molecules')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    
    print("GNN application completed!")