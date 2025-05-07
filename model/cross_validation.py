"""
Cross-validation module for molecular property prediction.
Provides functions for k-fold cross-validation of GNN models.
"""

import torch
import numpy as np
from sklearn.model_selection import KFold
import time
import os
from torch_geometric.data import Batch
from sklearn.metrics import r2_score


def perform_cross_validation(model_class, train_dataset, n_folds=2, epochs=10, n_repetitions=3, device='cpu', 
                             learning_rate=0.001, batch_size=64, output_dir=None, model_params=None):
    """
    Perform k-fold cross-validation on the provided dataset.
    
    Args:
        model_class: Class of the model to train (will be instantiated for each fold)
        train_dataset: Dataset to use for training and validation (list of PyG Data objects)
        n_folds: Number of folds for cross-validation
        epochs: Number of epochs per repetition
        n_repetitions: Number of repetitions (default=3)
        device: Device to use for training ('cpu' or 'cuda')
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        output_dir: Directory to save results
        model_params: Dictionary of parameters to pass to model constructor
        
    Returns:
        Dictionary with cross-validation results
    """
    # Initialize model_params if None
    if model_params is None:
        model_params = {}
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    fold_models = []
    
    # Convert dataset to numpy indices for splitting
    indices = np.arange(len(train_dataset))
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nTraining fold {fold+1}/{n_folds}...")
        
        # Get training and validation datasets for this fold
        X_train = [train_dataset[i] for i in train_idx]
        X_val = [train_dataset[i] for i in val_idx]
        
        # Extract targets
        y_train = [data.y.item() if hasattr(data, 'y') and data.y is not None else 0.0 for data in X_train]
        y_val = [data.y.item() if hasattr(data, 'y') and data.y is not None else 0.0 for data in X_val]
        
        # Initialize a new model for this fold using the provided parameters
        model = model_class(**model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        # Train with n_repetitions of epochs
        for rep in range(n_repetitions):
            print(f"  Repetition {rep+1}/{n_repetitions}:")
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                epoch_loss = 0
                
                # Process in batches
                indices = np.random.permutation(len(X_train))
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_data = [X_train[j] for j in batch_indices]
                    batch_targets = [y_train[j] for j in batch_indices]
                    
                    # Create batch
                    batch = Batch.from_data_list(batch_data)
                    batch_targets = torch.tensor(batch_targets, dtype=torch.float).view(-1, 1)
                    batch = batch.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    pred = model(batch)
                    loss = loss_fn(pred, batch_targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * len(batch_indices)
                
                train_loss = epoch_loss / len(train_idx)
                train_losses.append(train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for i in range(0, len(val_idx), batch_size):
                        batch_indices = np.arange(i, min(i+batch_size, len(val_idx)))
                        batch_data = [X_val[j-i] for j in batch_indices]
                        batch_targets = [y_val[j-i] for j in batch_indices]
                        
                        # Create batch
                        batch = Batch.from_data_list(batch_data)
                        batch_targets = torch.tensor(batch_targets, dtype=torch.float).view(-1, 1)
                        batch = batch.to(device)
                        batch_targets = batch_targets.to(device)
                        
                        pred = model(batch)
                        val_loss += loss_fn(pred, batch_targets).item() * len(batch_indices)
                
                val_loss = val_loss / len(val_idx)
                val_losses.append(val_loss)
                
                # Print only at the end of each repetition to reduce output
                if (epoch + 1) == epochs:
                    print(f"    Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        
        # Final evaluation on validation set
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in zip(X_val, y_val):
                data = data.to(device)
                pred = model(data).item()
                val_predictions.append(pred)
                val_targets.append(target)
        
        # Calculate metrics
        val_mse = np.mean((np.array(val_predictions) - np.array(val_targets)) ** 2)
        val_rmse = np.sqrt(val_mse)
        val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_targets)))
        val_r2 = r2_score(val_targets, val_predictions)
        
        fold_result = {
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'val_mse': val_mse,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2
        }
        fold_results.append(fold_result)
        fold_models.append(model)
    
    # Calculate average metrics across folds
    avg_val_rmse = np.mean([r['val_rmse'] for r in fold_results])
    avg_val_mae = np.mean([r['val_mae'] for r in fold_results])
    avg_val_r2 = np.mean([r['val_r2'] for r in fold_results])
    
    total_time = time.time() - start_time
    
    cv_results = {
        'fold_results': fold_results,
        'avg_val_rmse': avg_val_rmse,
        'avg_val_mae': avg_val_mae,
        'avg_val_r2': avg_val_r2,
        'total_time': total_time,
        'models': fold_models
    }
    
    # Write summary to file
    if output_dir:
        with open(os.path.join(output_dir, 'cross_validation_summary.txt'), 'w') as f:
            f.write(f"2-FOLD CROSS-VALIDATION RESULTS ({n_repetitions}x{epochs} epochs)\n")
            f.write(f"================================================\n\n")
            
            for fold, result in enumerate(fold_results):
                f.write(f"Fold {fold+1}:\n")
                f.write(f"  Best Validation Loss: {result['best_val_loss']:.4f}\n")
                f.write(f"  Validation RMSE: {result['val_rmse']:.4f}\n")
                f.write(f"  Validation MAE: {result['val_mae']:.4f}\n")
                f.write(f"  Validation R²: {result['val_r2']:.4f}\n\n")
            
            f.write(f"Average across folds:\n")
            f.write(f"  Validation RMSE: {avg_val_rmse:.4f}\n")
            f.write(f"  Validation MAE: {avg_val_mae:.4f}\n")
            f.write(f"  Validation R²: {avg_val_r2:.4f}\n\n")
            f.write(f"Total cross-validation time: {total_time:.2f} seconds\n")
    
    return cv_results
