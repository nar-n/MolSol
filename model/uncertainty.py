"""
Uncertainty Quantification for Molecular Property Prediction
This module provides methods for quantifying uncertainty in ML model predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class UncertaintyQuantifier:
    """Class for quantifying uncertainty in molecular property predictions"""
    
    @staticmethod
    def mc_dropout_uncertainty(model, data, n_samples=30):
        """
        Use Monte Carlo Dropout to estimate prediction uncertainty (Bayesian approach)
        
        Args:
            model: Trained MoleculeGNN model with dropout
            data: PyTorch Geometric Data object
            n_samples: Number of forward passes with dropout enabled
            
        Returns:
            mean_pred: Mean prediction
            std_pred: Standard deviation (epistemic uncertainty)
            ci_lower, ci_upper: 95% confidence interval
        """
        # Enable dropout in evaluation mode by subclassing the model and overriding forward
        class MCDropoutModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
            def forward(self, x):
                # Set encoder layers to train mode to enable dropout
                self.base_model.encoder.train()
                self.base_model.predictor.train()
                return self.base_model(x)
        
        mc_model = MCDropoutModel(model)
        mc_model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = mc_model(data).item()
                predictions.append(pred)
        
        # Calculate statistics
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # 95% confidence interval assuming normally distributed predictions
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        return mean_pred, std_pred, ci_lower, ci_upper, predictions
    
    @staticmethod
    def residual_based_uncertainty(model, data, residual_std):
        """
        Use residual-based uncertainty estimation (frequentist approach)
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric Data object
            residual_std: Standard deviation of residuals on known data
            
        Returns:
            pred: Prediction
            ci_lower, ci_upper: 95% confidence interval
        """
        model.eval()
        
        with torch.no_grad():
            pred = model(data).item()
        
        # Calculate 95% confidence interval
        ci_lower = pred - 1.96 * residual_std
        ci_upper = pred + 1.96 * residual_std
        
        return pred, ci_lower, ci_upper
    
    @staticmethod
    def combined_uncertainty(model, data, residual_std, n_samples=30):
        """
        Combine Bayesian and frequentist uncertainty approaches
        
        Args:
            model: Trained GNN model with dropout
            data: PyTorch Geometric Data object
            residual_std: Standard deviation of residuals on known data
            n_samples: Number of forward passes with dropout enabled
            
        Returns:
            mean_pred: Mean prediction
            total_std: Combined standard deviation
            ci_lower, ci_upper: 95% confidence interval
        """
        # Get Bayesian uncertainty (epistemic)
        mean_pred, epistemic_std, _, _, _ = UncertaintyQuantifier.mc_dropout_uncertainty(
            model, data, n_samples)
        
        # Combine uncertainties (epistemic from MC dropout, aleatoric from residuals)
        # This is a simplification of total variance = epistemic variance + aleatoric variance
        total_variance = epistemic_std**2 + residual_std**2
        total_std = np.sqrt(total_variance)
        
        # Calculate 95% confidence interval
        ci_lower = mean_pred - 1.96 * total_std
        ci_upper = mean_pred + 1.96 * total_std
        
        return mean_pred, total_std, ci_lower, ci_upper, epistemic_std, residual_std
    
    @staticmethod
    def evaluate_uncertainty_calibration(predictions, uncertainties, actual_values, bins=10):
        """
        Evaluate how well-calibrated the uncertainty estimates are
        
        Args:
            predictions: List of predicted values
            uncertainties: List of uncertainty estimates (standard deviations)
            actual_values: List of actual values
            bins: Number of bins for calibration analysis
            
        Returns:
            calibration_scores: Dictionary with calibration metrics
        """
        # Calculate standardized residuals
        standardized_residuals = [(y_true - y_pred) / (uncertainty * 1.96) 
                                 for y_true, y_pred, uncertainty in 
                                 zip(actual_values, predictions, uncertainties)]
        
        # Count how many actual values fall within the predicted confidence intervals
        within_68_ci = sum(abs(r) < 1.0 for r in standardized_residuals) / len(standardized_residuals)
        within_95_ci = sum(abs(r) < 2.0 for r in standardized_residuals) / len(standardized_residuals)
        within_99_ci = sum(abs(r) < 3.0 for r in standardized_residuals) / len(standardized_residuals)
        
        # Ideal calibration would have 68%, 95%, and 99% of values within these intervals
        cal_error_68 = abs(within_68_ci - 0.68)
        cal_error_95 = abs(within_95_ci - 0.95)
        cal_error_99 = abs(within_99_ci - 0.99)
        
        # Calculate overall calibration error
        calibration_error = (cal_error_68 + cal_error_95 + cal_error_99) / 3
        
        return {
            'within_68_ci': within_68_ci,
            'within_95_ci': within_95_ci,
            'within_99_ci': within_99_ci,
            'cal_error_68': cal_error_68,
            'cal_error_95': cal_error_95,
            'cal_error_99': cal_error_99,
            'calibration_error': calibration_error,
            'standardized_residuals': standardized_residuals
        }
    
    @staticmethod
    def plot_uncertainty_calibration(predictions, uncertainties, actual_values, output_dir=None):
        """
        Create calibration plots for uncertainty estimates
        
        Args:
            predictions: List of predicted values
            uncertainties: List of uncertainty estimates (standard deviations)
            actual_values: List of actual values
            output_dir: Directory to save plots
        """
        # Calculate standardized residuals
        standardized_residuals = [(y_true - y_pred) / (uncertainty * 1.96) 
                                 for y_true, y_pred, uncertainty in 
                                 zip(actual_values, predictions, uncertainties)]
        
        # Plot histogram of standardized residuals
        plt.figure(figsize=(10, 6))
        plt.hist(standardized_residuals, bins=30, alpha=0.7, density=True)
        plt.title('Histogram of Standardized Residuals')
        plt.xlabel('Standardized Residual')
        plt.ylabel('Density')
        # Add standard normal distribution for comparison
        x = np.linspace(-4, 4, 100)
        plt.plot(x, np.exp(-(x**2)/2) / np.sqrt(2*np.pi), 'r-', lw=2)
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'uncertainty_calibration_histogram.png'))
        else:
            plt.savefig('uncertainty_calibration_histogram.png')
        plt.close()
        
        # Plot uncertainty vs error
        errors = [abs(y_true - y_pred) for y_true, y_pred in zip(actual_values, predictions)]
        plt.figure(figsize=(10, 6))
        plt.scatter(uncertainties, errors, alpha=0.6)
        plt.title('Prediction Error vs Predicted Uncertainty')
        plt.xlabel('Predicted Uncertainty (std)')
        plt.ylabel('Absolute Error')
        
        # Add ideal calibration line
        max_val = max(max(uncertainties), max(errors))
        plt.plot([0, max_val], [0, max_val*1.96], 'r--', label='Ideal 95% CI calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'uncertainty_vs_error.png'))
        else:
            plt.savefig('uncertainty_vs_error.png')
        plt.close()
        
        # Create calibration curve
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = [uncertainties[i] for i in sorted_indices]
        sorted_errors = [errors[i] for i in sorted_indices]
        
        # Create bins and calculate average error in each bin
        n_bins = 10
        bin_size = len(sorted_uncertainties) // n_bins
        bin_avg_uncertainties = []
        bin_avg_errors = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_uncertainties)
            
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            bin_errors = sorted_errors[start_idx:end_idx]
            
            bin_avg_uncertainties.append(np.mean(bin_uncertainties))
            bin_avg_errors.append(np.mean(bin_errors))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(bin_avg_uncertainties, bin_avg_errors, s=100, alpha=0.7)
        plt.title('Calibration Curve: Mean Error vs Mean Uncertainty')
        plt.xlabel('Mean Predicted Uncertainty (std)')
        plt.ylabel('Mean Absolute Error')
        
        # Add ideal calibration line
        max_val = max(max(bin_avg_uncertainties), max(bin_avg_errors))
        plt.plot([0, max_val], [0, max_val*1.96], 'r--', label='Ideal 95% CI calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'uncertainty_calibration_curve.png'))
        else:
            plt.savefig('uncertainty_calibration_curve.png')
        plt.close()
