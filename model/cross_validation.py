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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def perform_cross_validation(model_class, train_dataset, n_folds=2, 
                             max_epochs=50, patience=10, min_epochs=20,
                             device='cpu', learning_rate=0.001, 
                             batch_size=64, output_dir=None, model_params=None,
                             scheduler_factor=0.5, scheduler_patience=5):
    """
    Perform k-fold cross-validation on the provided dataset with early stopping.
    
    Args:
        model_class: Class of the model to train
        train_dataset: List of PyG Data objects
        n_folds: Number of folds for cross-validation
        max_epochs: Maximum number of epochs
        patience: Early stopping patience (stop after N epochs without improvement)
        min_epochs: Minimum epochs before early stopping can take effect
        device: Device to use for training
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        output_dir: Directory to save results
        model_params: Dictionary of parameters for model constructor
        scheduler_factor: Factor for learning rate scheduler
        scheduler_patience: Patience for learning rate scheduler
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, 
            patience=scheduler_patience, verbose=True
        )
        loss_fn = torch.nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_count = 0
        best_epoch = 0
        
        print(f"  Training for up to {max_epochs} epochs with early stopping (patience={patience})")
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            epoch_loss = 0
            
            # Process in batches with shuffling
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
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            
            train_loss = epoch_loss / len(train_idx)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            
            with torch.no_grad():
                for i in range(0, len(val_idx), batch_size):
                    batch_indices = list(range(i, min(i+batch_size, len(X_val))))
                    batch_data = [X_val[j] for j in batch_indices]
                    batch_targets = [y_val[j] for j in batch_indices]
                    
                    # Create batch
                    batch = Batch.from_data_list(batch_data)
                    batch_targets = torch.tensor(batch_targets, dtype=torch.float).view(-1, 1)
                    batch = batch.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    pred = model(batch)
                    val_preds.extend(pred.cpu().numpy().flatten())
                    val_loss += loss_fn(pred, batch_targets).item() * len(batch_indices)
            
            val_loss = val_loss / len(val_idx)
            val_losses.append(val_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                train_r2 = r2_score(y_train, predict_on_dataset(model, X_train, device))
                val_r2 = r2_score(y_val, predict_on_dataset(model, X_val, device))
                print(f"    Epoch {epoch+1}/{max_epochs}: Train Loss: {train_loss:.4f} (R²: {train_r2:.3f}), "
                      f"Val Loss: {val_loss:.4f} (R²: {val_r2:.3f})")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
                best_epoch = epoch
            else:
                no_improve_count += 1
            
            # Early stopping with minimum training epochs
            if epoch >= min_epochs and no_improve_count >= patience:
                print(f"    Early stopping at epoch {epoch+1} - no improvement for {patience} epochs "
                      f"(best epoch: {best_epoch+1})")
                break
        
        # Load best model for this fold
        model.load_state_dict(best_model_state)
        model.to(device)
        
        # Final evaluation on validation set
        model.eval()
        val_predictions = predict_on_dataset(model, X_val, device)
        
        # Calculate metrics
        val_mse = mean_squared_error(y_val, val_predictions)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        
        print(f"  Fold {fold+1} results - Best epoch: {best_epoch+1}")
        print(f"    RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
        
        fold_result = {
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch + 1,
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
    avg_best_epoch = np.mean([r['best_epoch'] for r in fold_results])
    
    total_time = time.time() - start_time
    
    cv_results = {
        'fold_results': fold_results,
        'avg_val_rmse': avg_val_rmse,
        'avg_val_mae': avg_val_mae,
        'avg_val_r2': avg_val_r2,
        'avg_best_epoch': avg_best_epoch,
        'total_time': total_time,
        'models': fold_models
    }
    
    # Generate summary and convergence plots
    if output_dir:
        write_cv_summary(cv_results, output_dir)
        plot_cv_convergence(fold_results, output_dir)
    
    return cv_results

def predict_on_dataset(model, dataset, device):
    """Make predictions on a dataset"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            pred = model(data)
            predictions.append(pred.cpu().item())
    
    return predictions

def write_cv_summary(cv_results, output_dir):
    """Write cross validation summary to file"""
    with open(os.path.join(output_dir, 'cross_validation_summary.txt'), 'w') as f:
        f.write(f"CROSS-VALIDATION RESULTS\n")
        f.write(f"=======================\n\n")
        
        for fold, result in enumerate(cv_results['fold_results']):
            f.write(f"Fold {fold+1}:\n")
            f.write(f"  Best Epoch: {result['best_epoch']}\n")
            f.write(f"  Best Validation Loss: {result['best_val_loss']:.4f}\n")
            f.write(f"  Validation RMSE: {result['val_rmse']:.4f}\n")
            f.write(f"  Validation MAE: {result['val_mae']:.4f}\n")
            f.write(f"  Validation R²: {result['val_r2']:.4f}\n\n")
        
        f.write(f"Average across folds:\n")
        f.write(f"  Best Epoch: {cv_results['avg_best_epoch']:.1f}\n")
        f.write(f"  Validation RMSE: {cv_results['avg_val_rmse']:.4f}\n")
        f.write(f"  Validation MAE: {cv_results['avg_val_mae']:.4f}\n")
        f.write(f"  Validation R²: {cv_results['avg_val_r2']:.4f}\n\n")
        f.write(f"Total cross-validation time: {cv_results['total_time']:.2f} seconds\n")

def plot_cv_convergence(fold_results, output_dir):
    """Generate plots for cross-validation convergence"""
    import matplotlib.pyplot as plt
    
    # Determine max epochs across all folds
    max_epochs = max([len(r['train_losses']) for r in fold_results])
    
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation losses for each fold
    for i, result in enumerate(fold_results):
        epochs = len(result['train_losses'])
        plt.subplot(2, 2, 1)
        plt.plot(range(1, epochs+1), result['train_losses'], label=f'Fold {i+1}')
        
        plt.subplot(2, 2, 2)
        plt.plot(range(1, epochs+1), result['val_losses'], label=f'Fold {i+1}')
        
        # Mark best epoch
        best_epoch = result['best_epoch']
        best_val_loss = result['val_losses'][best_epoch-1] if best_epoch <= len(result['val_losses']) else None
        if best_val_loss is not None:
            plt.subplot(2, 2, 2)
            plt.scatter([best_epoch], [best_val_loss], marker='o', color=f'C{i}', s=100, alpha=0.5)
    
    # Configure subplots
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Add a bar chart of R² values
    plt.subplot(2, 2, 3)
    r2_values = [r['val_r2'] for r in fold_results]
    fold_names = [f'Fold {i+1}' for i in range(len(fold_results))]
    plt.bar(fold_names, r2_values)
    plt.axhline(np.mean(r2_values), color='red', linestyle='--', label=f'Mean: {np.mean(r2_values):.3f}')
    plt.xlabel('Fold')
    plt.ylabel('Validation R²')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add a bar chart of best epochs
    plt.subplot(2, 2, 4)
    best_epochs = [r['best_epoch'] for r in fold_results]
    plt.bar(fold_names, best_epochs)
    plt.axhline(np.mean(best_epochs), color='red', linestyle='--', label=f'Mean: {np.mean(best_epochs):.1f}')
    plt.xlabel('Fold')
    plt.ylabel('Best Epoch')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_convergence.png'))
    plt.close()
