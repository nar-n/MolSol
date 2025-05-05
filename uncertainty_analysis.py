"""
Uncertainty Analysis Tool for Molecular Property Predictions

This script allows for standalone analysis of prediction uncertainties for any set of molecules.
It loads a pre-trained model and provides detailed uncertainty metrics and visualizations.

Features:
- Load pre-trained GNN models
- Predict properties with uncertainty quantification
- Generate uncertainty calibration plots
- Identify molecules with highest/lowest uncertainty
- Compare different uncertainty estimation methods
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
from collections import defaultdict

# Import from main script
from main import MolecularGNN, MoleculeGraphConverter, UncertaintyQuantifier

def load_pretrained_model(model_path, feature_dim=14, physchem_dim=13):
    """Load a pretrained GNN model"""
    # Create model with the same architecture
    model = MolecularGNN(in_features=feature_dim, hidden_features=128, latent_dim=64, 
                         physchem_features=physchem_dim, out_features=1)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def analyze_uncertainty_for_molecules(model, smiles_list, output_dir=None, known_values=None, n_mc_samples=50):
    """
    Analyze uncertainty for a list of SMILES
    
    Args:
        model: Trained GNN model
        smiles_list: List of SMILES strings
        output_dir: Directory to save results
        known_values: Optional list of known property values for comparison
        n_mc_samples: Number of Monte Carlo samples for Bayesian uncertainty
        
    Returns:
        DataFrame with predictions and uncertainty metrics
    """
    converter = MoleculeGraphConverter()
    
    results = []
    valid_smiles = []
    failed_smiles = []
    
    # For uncertainty calibration if known values are provided
    predictions = []
    uncertainties = []
    actual_values = []
    
    print(f"Analyzing uncertainty for {len(smiles_list)} molecules...")
    
    # Use fixed residual std for the frequentist approach
    residual_std = 0.6  # This should ideally come from validation data
    
    for i, smiles in enumerate(smiles_list):
        if i % 50 == 0 and i > 0:
            print(f"Processed {i}/{len(smiles_list)} molecules...")
        
        try:
            mol_graph = converter.smiles_to_graph(smiles)
            if mol_graph:
                # Get features
                features = mol_graph['node_features']
                adj = mol_graph['adj_matrix']
                physchem = mol_graph['physchem_features']
                mol = mol_graph['mol']
                
                # Get Bayesian uncertainty
                mean_pred_bayes, epistemic_std, ci_lower_bayes, ci_upper_bayes, mc_samples = (
                    UncertaintyQuantifier.mc_dropout_uncertainty(
                        model, features, adj, physchem, n_samples=n_mc_samples
                    )
                )
                
                # Get frequentist uncertainty
                mean_pred_freq, ci_lower_freq, ci_upper_freq = (
                    UncertaintyQuantifier.residual_based_uncertainty(
                        model, features, adj, physchem, residual_std
                    )
                )
                
                # Get combined uncertainty
                mean_pred_combined, total_std, ci_lower_combined, ci_upper_combined, _, _ = (
                    UncertaintyQuantifier.combined_uncertainty(
                        model, features, adj, physchem, residual_std, n_samples=n_mc_samples
                    )
                )
                
                # Save result
                result = {
                    'smiles': smiles,
                    'prediction': mean_pred_combined,
                    'total_uncertainty': total_std,
                    'epistemic_uncertainty': epistemic_std,
                    'aleatoric_uncertainty': residual_std,
                    'ci_lower_95': ci_lower_combined,
                    'ci_upper_95': ci_upper_combined,
                    'bayes_prediction': mean_pred_bayes,
                    'freq_prediction': mean_pred_freq,
                    'mc_sample_std': np.std(mc_samples),
                    'prediction_variance': np.var(mc_samples),
                    'sample_count': n_mc_samples
                }
                
                # Add known value if provided
                if known_values is not None and i < len(known_values):
                    result['actual_value'] = known_values[i]
                    result['error'] = abs(known_values[i] - mean_pred_combined)
                    
                    # For uncertainty calibration
                    predictions.append(mean_pred_combined)
                    uncertainties.append(total_std)
                    actual_values.append(known_values[i])
                
                results.append(result)
                valid_smiles.append(smiles)
            else:
                failed_smiles.append((smiles, "Could not convert to graph"))
        except Exception as e:
            failed_smiles.append((smiles, str(e)))
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    print(f"Successfully processed {len(valid_smiles)} molecules. Failed: {len(failed_smiles)}")
    
    # Generate uncertainty calibration if known values are provided
    if known_values is not None and len(actual_values) > 0:
        print("\nEvaluating uncertainty calibration...")
        calibration_metrics = UncertaintyQuantifier.evaluate_uncertainty_calibration(
            predictions, uncertainties, actual_values)
        
        print(f"% within 68% CI: {calibration_metrics['within_68_ci']:.2f} (ideal: 0.68)")
        print(f"% within 95% CI: {calibration_metrics['within_95_ci']:.2f} (ideal: 0.95)")
        print(f"Calibration error: {calibration_metrics['calibration_error']:.4f} (lower is better)")
        
        if output_dir:
            UncertaintyQuantifier.plot_uncertainty_calibration(
                predictions, uncertainties, actual_values, output_dir)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save DataFrame to CSV
        df.to_csv(os.path.join(output_dir, 'uncertainty_analysis.csv'), index=False)
        
        # Generate visualizations
        if len(df) > 0:
            # 1. Distribution of uncertainties
            plt.figure(figsize=(10, 6))
            plt.hist(df['total_uncertainty'], bins=20, alpha=0.7)
            plt.axvline(df['total_uncertainty'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["total_uncertainty"].mean():.4f}')
            plt.axvline(df['total_uncertainty'].median(), color='green', linestyle='--', 
                       label=f'Median: {df["total_uncertainty"].median():.4f}')
            plt.title('Distribution of Prediction Uncertainties')
            plt.xlabel('Uncertainty (std)')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'))
            plt.close()
            
            # 2. Scatterplot of prediction vs uncertainty
            plt.figure(figsize=(10, 6))
            plt.scatter(df['prediction'], df['total_uncertainty'], alpha=0.7)
            plt.title('Prediction Value vs. Uncertainty')
            plt.xlabel('Predicted Value')
            plt.ylabel('Uncertainty (std)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'prediction_vs_uncertainty.png'))
            plt.close()
            
            # 3. Uncertainty component breakdown
            plt.figure(figsize=(10, 6))
            plt.bar(['Epistemic (Model)', 'Aleatoric (Data)'], 
                   [df['epistemic_uncertainty'].mean(), df['aleatoric_uncertainty'].mean()],
                  yerr=[df['epistemic_uncertainty'].std(), df['aleatoric_uncertainty'].std()],
                  capsize=10, alpha=0.7)
            plt.title('Components of Prediction Uncertainty')
            plt.ylabel('Average Uncertainty (std)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'uncertainty_components.png'))
            plt.close()
            
            # 4. Draw molecules with highest/lowest uncertainty if RDKit available
            try:
                # Top 5 highest uncertainty
                highest_unc = df.nlargest(5, 'total_uncertainty')
                highest_mols = [Chem.MolFromSmiles(s) for s in highest_unc['smiles']]
                highest_mols = [m for m in highest_mols if m is not None]
                
                if highest_mols:
                    img = Draw.MolsToGridImage(
                        highest_mols, 
                        molsPerRow=min(5, len(highest_mols)),
                        subImgSize=(300, 300),
                        legends=[f"{row.smiles}\nPred: {row.prediction:.2f} ± {row.total_uncertainty:.2f}" 
                                for _, row in highest_unc.iterrows() if Chem.MolFromSmiles(row.smiles)]
                    )
                    img.save(os.path.join(output_dir, 'highest_uncertainty_molecules.png'))
                
                # Top 5 lowest uncertainty
                lowest_unc = df.nsmallest(5, 'total_uncertainty')
                lowest_mols = [Chem.MolFromSmiles(s) for s in lowest_unc['smiles']]
                lowest_mols = [m for m in lowest_mols if m is not None]
                
                if lowest_mols:
                    img = Draw.MolsToGridImage(
                        lowest_mols, 
                        molsPerRow=min(5, len(lowest_mols)),
                        subImgSize=(300, 300),
                        legends=[f"{row.smiles}\nPred: {row.prediction:.2f} ± {row.total_uncertainty:.2f}" 
                                for _, row in lowest_unc.iterrows() if Chem.MolFromSmiles(row.smiles)]
                    )
                    img.save(os.path.join(output_dir, 'lowest_uncertainty_molecules.png'))
            except Exception as e:
                print(f"Warning: Could not generate molecule images: {e}")
    
    return df, failed_smiles

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction uncertainty for molecules')
    parser.add_argument('--model', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--input', type=str, required=True, help='CSV file with SMILES column')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='Name of SMILES column')
    parser.add_argument('--actual_col', type=str, help='Name of column with actual values (optional)')
    parser.add_argument('--output', type=str, default='uncertainty_results', help='Output directory')
    parser.add_argument('--mc_samples', type=int, default=50, help='Number of Monte Carlo samples')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_pretrained_model(args.model)
    
    # Load data
    print(f"Loading molecules from {args.input}...")
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_col].tolist()
    
    # Get known values if available
    known_values = None
    if args.actual_col and args.actual_col in df.columns:
        known_values = df[args.actual_col].tolist()
        print(f"Found {len(known_values)} known values for comparison")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Analyze uncertainty
    results_df, failed = analyze_uncertainty_for_molecules(
        model, smiles_list, args.output, known_values, args.mc_samples)
    
    print(f"Results saved to {args.output}")
    print(f"Summary: processed {len(results_df)} molecules, failed {len(failed)}")
    
    if len(results_df) > 0:
        print("\nUncertainty statistics:")
        print(f"Average uncertainty (std): {results_df['total_uncertainty'].mean():.4f}")
        print(f"Min uncertainty: {results_df['total_uncertainty'].min():.4f}")
        print(f"Max uncertainty: {results_df['total_uncertainty'].max():.4f}")
        
        if 'error' in results_df.columns:
            corr = np.corrcoef(results_df['total_uncertainty'], results_df['error'])[0, 1]
            print(f"\nCorrelation between uncertainty and error: {corr:.4f}")
            print("(Positive values indicate well-calibrated uncertainty)")

if __name__ == "__main__":
    main()
