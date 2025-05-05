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
try:
    from rdkit.Chem import QED
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
from torch_geometric.data import Data, Batch

# Import functionality from other modules
from model.smiles_to_graph import SMILESToGraph
from model.gnn_encoder import GNNEncoder
from model.property_predictor import MoleculeGNN
from model.uncertainty import UncertaintyQuantifier  # Import from the new file
from model.feature_importance import export_feature_importance_to_csv

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
            input_dim = X_train_fold[0].x.shape[1]
            model = MoleculeGNN(input_dim=input_dim, hidden_dim=128, latent_dim=64, n_tasks=1, gnn_type='GCN')
            
            # Train model
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
            loss_fn = torch.nn.MSELoss()
            
            # Simple training loop for CV
            for epoch in range(epochs):
                # Training
                model.train()
                
                # Create batches
                train_batches = [X_train_fold[i:i+batch_size] for i in range(0, len(X_train_fold), batch_size)]
                train_targets = [y_train_fold[i:i+batch_size] for i in range(0, len(y_train_fold), batch_size)]
                
                for batch, targets in zip(train_batches, train_targets):
                    batch_data = Batch.from_data_list(batch)
                    
                    optimizer.zero_grad()
                    pred = model(batch_data)
                    targets_tensor = torch.tensor(targets, dtype=torch.float).view(-1, 1)
                    loss = loss_fn(pred, targets_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Print progress at specific intervals
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}")
            
            # Evaluate on validation set
            model.eval()
            fold_preds = []
            
            with torch.no_grad():
                # Process one by one to match the expected API
                for data, target in zip(X_val_fold, y_val_fold):
                    pred = model(data).item()
                    fold_preds.append(pred)
                    
                    all_predictions.append(pred)
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

# %% Load ESOL Dataset
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

# %% Predict Unseen Molecules
def predict_unseen_molecules(model, prediction_df):
    """Predict solubility for unseen molecules with uncertainty quantification"""
    converter = SMILESToGraph()
    solubility_col = 'solubility'
    
    predicted_values = []
    actual_values = []
    smiles_list = []
    failed_molecules = []
    
    # For uncertainty quantification
    uncertainties = []
    confidence_intervals = []
    epistemic_uncertainties = []  # Model uncertainty
    aleatoric_uncertainties = []  # Data uncertainty
    
    print(f"Predicting solubility for {len(prediction_df)} unseen molecules with uncertainty quantification...")
    
    # First calculate residual standard deviation on the test set for frequentist approach
    # We'll use a fixed value for simplicity in this implementation
    residual_std = 0.6  # You could calculate this from training/validation data
    
    for _, row in prediction_df.iterrows():
        try:
            smiles = row['smiles']
            actual_solubility = row[solubility_col]
            
            # Convert SMILES to PyG data object
            data = converter.convert(smiles)
            if data is not None:
                # Make prediction with uncertainty
                mean_pred, total_std, ci_lower, ci_upper, epistemic_std, aleatoric_std = (
                    UncertaintyQuantifier.combined_uncertainty(
                        model, data, residual_std, n_samples=20
                    )
                )
                
                # Store results
                predicted_values.append(mean_pred)
                actual_values.append(actual_solubility)
                smiles_list.append(smiles)
                uncertainties.append(total_std)
                confidence_intervals.append((ci_lower, ci_upper))
                epistemic_uncertainties.append(epistemic_std)
                aleatoric_uncertainties.append(aleatoric_std)
            else:
                failed_molecules.append((smiles, "Could not convert to graph"))
        except Exception as e:
            failed_molecules.append((smiles, str(e)))
    
    print(f"Successfully predicted {len(predicted_values)} molecules with uncertainty")
    print(f"Failed to predict {len(failed_molecules)} molecules")
    
    # Sample of predictions with uncertainty
    print("\nSample predictions with uncertainty:")
    for i in range(min(5, len(predicted_values))):
        print(f"Molecule: {smiles_list[i]}")
        print(f"  Actual: {actual_values[i]:.4f}, Predicted: {predicted_values[i]:.4f}, "
              f"Error: {abs(actual_values[i] - predicted_values[i]):.4f}")
        print(f"  Uncertainty (std): {uncertainties[i]:.4f}, 95% CI: [{confidence_intervals[i][0]:.4f}, {confidence_intervals[i][1]:.4f}]")
    
    # Evaluate uncertainty calibration
    calibration_metrics = UncertaintyQuantifier.evaluate_uncertainty_calibration(
        predicted_values, uncertainties, actual_values)
    
    print("\nUncertainty calibration metrics:")
    print(f"% within 68% CI: {calibration_metrics['within_68_ci']:.2f} (ideal: 0.68)")
    print(f"% within 95% CI: {calibration_metrics['within_95_ci']:.2f} (ideal: 0.95)")
    print(f"% within 99% CI: {calibration_metrics['within_99_ci']:.2f} (ideal: 0.99)")
    print(f"Calibration error: {calibration_metrics['calibration_error']:.4f} (lower is better)")
    
    # Calculate standard metrics
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
        'uncertainties': uncertainties,
        'confidence_intervals': confidence_intervals,
        'epistemic_uncertainties': epistemic_uncertainties,
        'aleatoric_uncertainties': aleatoric_uncertainties,
        'calibration_metrics': calibration_metrics
    }

# %% Train Solubility Model
def train_solubility_model(train_df, epochs=300, batch_size=32, output_dir=None):
    """Train the GNN model on training data"""
    converter = SMILESToGraph()
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
            
            data = converter.convert(smiles)
            if data is not None:
                data.y = torch.tensor([solubility], dtype=torch.float)
                X.append(data)
                y.append(solubility)
            else:
                failed_count += 1
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
    input_dim = X_train[0].x.shape[1]
    model = MoleculeGNN(input_dim=input_dim, hidden_dim=128, latent_dim=64, n_tasks=1, gnn_type='GCN')
    
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
        
        # Create batches
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = [X_train[i] for i in indices]
        y_train_shuffled = [y_train[i] for i in indices]
        
        # Process in batches
        for i in range(0, len(X_train_shuffled), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            # Create batch
            batch_data = Batch.from_data_list(batch_X)
            batch_targets = torch.tensor(batch_y, dtype=torch.float).view(-1, 1)
            
            optimizer.zero_grad()
            pred = model(batch_data)
            loss = loss_fn(pred, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(batch_X)
        
        train_loss /= len(X_train)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in zip(X_val, y_val):
                pred = model(data)
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
            best_model_state = copy.deepcopy(model.state_dict())
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
        for data, target in zip(X_train, y_train):
            pred = model(data).item()
            train_preds.append(pred)
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
        for data, target in zip(X_test, y_test):
            pred = model(data).item()
            test_preds.append(pred)
            test_targets.append(target)
    
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(test_targets, test_preds)),
        'mae': mean_absolute_error(test_targets, test_preds),
        'r2': r2_score(test_targets, test_preds)
    }
    
    # Load best model
    model.load_state_dict(best_model_state)
    
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
    plt.close()
    
    return model, train_metrics, test_metrics, best_model_state, X_test, y_test

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
    print("\n3. Training model on ESOL dataset with our modular GNN architecture...")
    model, train_metrics, test_metrics, best_model_state, X_test, y_test = train_solubility_model(train_df, output_dir=output_dir)
    
    # Step 4: Evaluate performance
    print("\n4. Evaluating model performance")
    print(f"Training metrics: RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, R²={train_metrics['r2']:.4f}")
    print(f"Test metrics: RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}, R²={test_metrics['r2']:.4f}")
    
    # Step 5: Advanced model evaluation - Feature importance
    print("\n5. Analyzing feature importance")
    feature_imp_df = export_feature_importance_to_csv(model, X_test, y_test, output_dir)
    
    print("\nTop 5 most important features:")
    for idx, row in feature_imp_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']} ({row['Category']}): {row['Importance_Score']:.4f}")
    
    # Step 6: Predict solubility for unseen molecules with uncertainty
    print("\n6. Predicting solubility for unseen molecules with uncertainty quantification")
    prediction_results = predict_unseen_molecules(model, prediction_df)
    
    # Generate uncertainty calibration plots
    print("\nGenerating uncertainty calibration plots...")
    UncertaintyQuantifier.plot_uncertainty_calibration(
        prediction_results['predicted'],
        prediction_results['uncertainties'],
        prediction_results['actual'],
        output_dir=output_dir
    )
    
    # Step 7: Final results
    print("\n7. Final results")
    print(f"Unseen prediction set metrics:")
    print(f"  RMSE: {prediction_results['rmse']:.4f}")
    print(f"  MAE: {prediction_results['mae']:.4f}")
    print(f"  R²: {prediction_results['r2']:.4f}")
    print(f"  Uncertainty calibration error: {prediction_results['calibration_metrics']['calibration_error']:.4f}")
    
    # Generate prediction visualization with uncertainty
    print("\nGenerating unseen predictions visualization with uncertainty...")
    plt.figure(figsize=(10, 8))
    
    # Calculate error bars for uncertainty visualization (95% confidence intervals)
    error_bars = [1.96 * u for u in prediction_results['uncertainties']]
    
    # Create scatter plot with error bars
    plt.errorbar(
        prediction_results['actual'],
        prediction_results['predicted'],
        yerr=error_bars,
        fmt='none',
        ecolor='lightgray',
        alpha=0.5,
        capsize=3
    )
    
    # Add main scatter points colored by uncertainty
    scatter = plt.scatter(
        prediction_results['actual'],
        prediction_results['predicted'],
        c=prediction_results['uncertainties'],
        cmap='viridis',
        alpha=0.8,
        s=80,
        edgecolors='k',
        linewidths=0.5
    )
    
    # Add identity line (perfect predictions)
    min_val = min(min(prediction_results['actual']), min(prediction_results['predicted']))
    max_val = max(max(prediction_results['actual']), max(prediction_results['predicted']))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    
    # Add colorbar for uncertainty magnitude
    cbar = plt.colorbar(scatter)
    cbar.set_label('Prediction Uncertainty (std)', fontsize=12)
    
    # Add labels and title
    plt.xlabel('Experimental LogS', fontsize=14)
    plt.ylabel('Predicted LogS', fontsize=14)
    plt.title('Predicted vs Experimental Solubility with Uncertainty', fontsize=16)
    
    # Add RMSE and R² info as text
    plt.text(
        min_val + 0.05 * (max_val - min_val),
        max_val - 0.1 * (max_val - min_val),
        f"RMSE: {prediction_results['rmse']:.3f}\nR²: {prediction_results['r2']:.3f}",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Improve appearance
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the visualization
    vis_path = os.path.join(output_dir, 'unseen_predictions_with_uncertainty.png')
    plt.savefig(vis_path, dpi=300)
    plt.close()
    print(f"Visualization saved to: {vis_path}")
    
    # Save model state dictionary
    torch.save(best_model_state, os.path.join(output_dir, 'best_model.pt'))
    
    # Generate comprehensive report file
    print("\nGenerating comprehensive evaluation report...")
    with open(os.path.join(output_dir, 'advanced_evaluation_results.txt'), 'w') as f:
        f.write(f"GNNSol Molecular Solubility Prediction Results with Uncertainty Quantification\n")
        f.write(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add model architecture and feature information
        f.write(f"Model Architecture:\n")
        f.write(f"  Combined Graph Neural Network with PyTorch Geometric\n")
        f.write(f"  Encoder: GNN layers with residual connections and global pooling\n")
        f.write(f"  Uncertainty quantification using Monte Carlo Dropout (Bayesian) and residual-based (frequentist) approaches\n\n")
        
        f.write(f"Molecular Graph Features:\n")
        f.write(f"  - Atoms: H, C, N, O, F, P, S, Cl, Br, I (one-hot encoded)\n")
        f.write(f"  - Formal charge\n")
        f.write(f"  - Radical electrons\n")
        f.write(f"  - Aromaticity\n")
        f.write(f"  - Number of attached hydrogens\n\n")
        
        f.write(f"Training metrics:\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {train_metrics['mae']:.4f}\n")
        f.write(f"  R²: {train_metrics['r2']:.4f}\n\n")
        
        f.write(f"Test metrics:\n")
        f.write(f"  RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"  MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"  R²: {test_metrics['r2']:.4f}\n\n")
        
        # Feature importance
        f.write(f"Feature Importance (Top 5):\n")
        for i, row in enumerate(feature_imp_df.head(5).itertuples(index=False)):
            f.write(f"  {i+1}. {row.Feature} ({row.Category}): {row.Importance_Score:.4f}\n")
        f.write("\n")
        
        # Prediction results on unseen data
        f.write(f"Unseen prediction set metrics:\n")
        f.write(f"  RMSE: {prediction_results['rmse']:.4f}\n")
        f.write(f"  MAE: {prediction_results['mae']:.4f}\n")
        f.write(f"  R²: {prediction_results['r2']:.4f}\n\n")
        
        # Uncertainty calibration metrics
        f.write(f"Uncertainty Calibration Metrics:\n")
        f.write(f"  % within 68% CI: {prediction_results['calibration_metrics']['within_68_ci']:.2f} (ideal: 0.68)\n")
        f.write(f"  % within 95% CI: {prediction_results['calibration_metrics']['within_95_ci']:.2f} (ideal: 0.95)\n")
        f.write(f"  % within 99% CI: {prediction_results['calibration_metrics']['within_99_ci']:.2f} (ideal: 0.99)\n")
        f.write(f"  Calibration error: {prediction_results['calibration_metrics']['calibration_error']:.4f}\n\n")
        
        # Sample predictions with uncertainty
        f.write(f"Sample predictions with uncertainty:\n")
        for i in range(min(5, len(prediction_results['smiles']))):
            f.write(f"Molecule: {prediction_results['smiles'][i]}\n")
            f.write(f"  Actual: {prediction_results['actual'][i]:.4f}, Predicted: {prediction_results['predicted'][i]:.4f}, ")
            f.write(f"Error: {abs(prediction_results['actual'][i] - prediction_results['predicted'][i]):.4f}\n")
            f.write(f"  Uncertainty (std): {prediction_results['uncertainties'][i]:.4f}, ")
            f.write(f"95% CI: [{prediction_results['confidence_intervals'][i][0]:.4f}, {prediction_results['confidence_intervals'][i][1]:.4f}]\n")
            f.write(f"  Epistemic uncertainty: {prediction_results['epistemic_uncertainties'][i]:.4f}, ")
            f.write(f"Aleatoric uncertainty: {prediction_results['aleatoric_uncertainties'][i]:.4f}\n\n")
    
    print(f"Comprehensive report saved to: {os.path.join(output_dir, 'advanced_evaluation_results.txt')}")
    
    total_time = time.time() - start_time
    print(f"\nSolubility prediction completed in {total_time:.1f} seconds")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
