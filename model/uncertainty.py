"""
Uncertainty Quantification for Molecular Property Prediction
This module provides methods for quantifying uncertainty in ML model predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

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
    def estimate_aleatoric_uncertainty(model, data, mc_predictions=None, n_samples=None):
        """
        Estimate aleatoric uncertainty (data noise) dynamically
        
        Args:
            model: Trained GNN model
            data: PyTorch Geometric Data object
            mc_predictions: Optional pre-computed MC dropout predictions
            n_samples: Number of samples if mc_predictions is not provided
            
        Returns:
            aleatoric_std: Estimated aleatoric uncertainty
        """
        # Use pre-computed predictions or generate new ones
        if mc_predictions is None:
            if n_samples is None:
                n_samples = 20
            
            _, _, _, _, mc_predictions = UncertaintyQuantifier.mc_dropout_uncertainty(
                model, data, n_samples)
        
        # Compute prediction mean
        pred_mean = np.mean(mc_predictions)
        
        # Dynamic aleatoric uncertainty based on prediction value
        # Higher uncertainty for more extreme values (very soluble or very insoluble)
        baseline_noise = 0.3  # Baseline noise level
        
        # Increase uncertainty for predictions away from the typical range (-2 to -4)
        if pred_mean > -2:
            # More soluble compounds tend to have higher measurement variation
            aleatoric_std = baseline_noise + 0.1 * abs(pred_mean + 2)
        elif pred_mean < -4:
            # Very insoluble compounds are harder to measure precisely
            aleatoric_std = baseline_noise + 0.05 * abs(pred_mean + 4)
        else:
            # Compounds in the typical range
            aleatoric_std = baseline_noise
            
        # Add small variance to ensure positive values
        aleatoric_std = max(aleatoric_std, 0.1)
        
        return aleatoric_std
    
    @staticmethod
    def combined_uncertainty(model, data, residual_std=None, n_samples=30):
        """
        Combine Bayesian and frequentist uncertainty approaches with dynamically estimated
        aleatoric uncertainty per sample
        
        Args:
            model: Trained GNN model with dropout
            data: PyTorch Geometric Data object
            residual_std: Optional fixed residual std (if None, will be estimated per sample)
            n_samples: Number of forward passes with dropout enabled
            
        Returns:
            mean_pred: Mean prediction
            total_std: Combined standard deviation
            ci_lower, ci_upper: 95% confidence interval
            epistemic_std: Model uncertainty
            aleatoric_std: Data uncertainty
        """
        # Get Bayesian uncertainty (epistemic) and MC predictions
        mean_pred, epistemic_std, _, _, mc_predictions = UncertaintyQuantifier.mc_dropout_uncertainty(
            model, data, n_samples)
        
        # Estimate aleatoric uncertainty dynamically for this sample
        if residual_std is None:
            aleatoric_std = UncertaintyQuantifier.estimate_aleatoric_uncertainty(
                model, data, mc_predictions)
        else:
            # Use provided residual_std as baseline but adjust based on prediction
            pred_abs = abs(mean_pred)
            # Increase uncertainty for more extreme predictions
            adjustment_factor = 1.0 + 0.1 * max(0, pred_abs - 3) 
            aleatoric_std = residual_std * adjustment_factor
        
        # Combine uncertainties (total variance = epistemic variance + aleatoric variance)
        total_variance = epistemic_std**2 + aleatoric_std**2
        total_std = np.sqrt(total_variance)
        
        # Calculate dynamic 95% confidence interval based on the combined uncertainty
        ci_lower = mean_pred - 1.96 * total_std
        ci_upper = mean_pred + 1.96 * total_std
        
        return mean_pred, total_std, ci_lower, ci_upper, epistemic_std, aleatoric_std
    
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
    
    @staticmethod
    def plot_uncertainty_components(predictions, epistemic_uncertainties, aleatoric_uncertainties, 
                                   actual_values=None, output_dir=None):
        """
        Create plots showing the contribution of epistemic and aleatoric uncertainty components
        
        Args:
            predictions: List of predicted values
            epistemic_uncertainties: List of epistemic uncertainty values (std)
            aleatoric_uncertainties: List of aleatoric uncertainty values (std)
            actual_values: Optional list of actual values
            output_dir: Directory to save plots
        """
        # Calculate total uncertainties
        total_uncertainties = [np.sqrt(e**2 + a**2) for e, a in 
                              zip(epistemic_uncertainties, aleatoric_uncertainties)]
        
        # Plot the relationship between prediction value and uncertainty components
        plt.figure(figsize=(12, 6))
        
        # Sort by prediction for cleaner visualization
        sorted_indices = np.argsort(predictions)
        sorted_preds = [predictions[i] for i in sorted_indices]
        sorted_epistemic = [epistemic_uncertainties[i] for i in sorted_indices]
        sorted_aleatoric = [aleatoric_uncertainties[i] for i in sorted_indices]
        sorted_total = [total_uncertainties[i] for i in sorted_indices]
        
        plt.plot(sorted_preds, sorted_epistemic, 'b-', alpha=0.7, label='Epistemic (Model Uncertainty)')
        plt.plot(sorted_preds, sorted_aleatoric, 'r-', alpha=0.7, label='Aleatoric (Data Uncertainty)')
        plt.plot(sorted_preds, sorted_total, 'k-', alpha=0.7, label='Total Uncertainty')
        
        plt.title('Uncertainty Components by Prediction Value')
        plt.xlabel('Predicted Solubility (LogS)')
        plt.ylabel('Uncertainty (Standard Deviation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'uncertainty_components.png'))
        else:
            plt.savefig('uncertainty_components.png')
        plt.close()
        
        # Stacked area chart of uncertainty contributions
        plt.figure(figsize=(12, 6))
        
        # Calculate percentage contributions
        epistemic_contrib = [e**2 / (e**2 + a**2) * 100 for e, a in 
                            zip(sorted_epistemic, sorted_aleatoric)]
        aleatoric_contrib = [a**2 / (e**2 + a**2) * 100 for e, a in 
                            zip(sorted_epistemic, sorted_aleatoric)]
        
        plt.stackplot(sorted_preds, [epistemic_contrib, aleatoric_contrib], 
                     labels=['Epistemic', 'Aleatoric'],
                     colors=['blue', 'red'], alpha=0.7)
        
        plt.title('Relative Contribution of Uncertainty Components')
        plt.xlabel('Predicted Solubility (LogS)')
        plt.ylabel('Contribution (%)')
        plt.legend()
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'uncertainty_contributions.png'))
        else:
            plt.savefig('uncertainty_contributions.png')
        plt.close()
        
        # If actual values are provided, plot error vs uncertainty components
        if actual_values is not None:
            errors = [abs(a - p) for a, p in zip(actual_values, predictions)]
            
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(epistemic_uncertainties, errors, alpha=0.6)
            plt.title('Error vs Epistemic Uncertainty')
            plt.xlabel('Epistemic Uncertainty')
            plt.ylabel('Absolute Error')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.scatter(aleatoric_uncertainties, errors, alpha=0.6)
            plt.title('Error vs Aleatoric Uncertainty')
            plt.xlabel('Aleatoric Uncertainty')
            plt.ylabel('Absolute Error')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.scatter(total_uncertainties, errors, alpha=0.6)
            plt.title('Error vs Total Uncertainty')
            plt.xlabel('Total Uncertainty')
            plt.ylabel('Absolute Error')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.scatter(epistemic_uncertainties, aleatoric_uncertainties, 
                       c=errors, cmap='viridis', alpha=0.6)
            plt.colorbar(label='Absolute Error')
            plt.title('Aleatoric vs Epistemic Uncertainty')
            plt.xlabel('Epistemic Uncertainty')
            plt.ylabel('Aleatoric Uncertainty')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'error_vs_uncertainty.png'))
            else:
                plt.savefig('error_vs_uncertainty.png')
            plt.close()
