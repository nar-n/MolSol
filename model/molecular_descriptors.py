"""
Molecular descriptors calculation for solubility prediction
This module computes key physicochemical properties that influence solubility
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf
import numpy as np
import torch

class MolecularDescriptors:
    """Calculate physicochemical descriptors important for solubility prediction"""
    
    @staticmethod
    def calculate_descriptors(mol):
        """
        Calculate key physicochemical descriptors for a molecule
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            descriptor_dict: Dictionary of computed descriptors
        """
        if mol is None:
            return None
            
        descriptors = {}
        
        try:
            # Lipophilicity - critical for solubility
            descriptors['logP'] = Crippen.MolLogP(mol)
            
            # Molecular size and weight
            descriptors['mol_weight'] = Descriptors.MolWt(mol)
            
            # Polarity descriptors
            descriptors['tpsa'] = MolSurf.TPSA(mol)
            
            # Hydrogen bonding
            descriptors['h_donors'] = Lipinski.NumHDonors(mol)
            descriptors['h_acceptors'] = Lipinski.NumHAcceptors(mol)
            
            # Molecular flexibility
            descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            
            # Molecular shape and volume
            descriptors['mol_refractivity'] = Crippen.MolMR(mol)
            
            # Surface area features (approximation)
            descriptors['sasa'] = MolSurf.LabuteASA(mol)
            
            # Ring characteristics
            descriptors['aromatic_rings'] = Lipinski.NumAromaticRings(mol)
            descriptors['aliphatic_rings'] = Lipinski.NumAliphaticRings(mol)
            
            # General Solubility Equation features
            descriptors['melting_point_estimate'] = 0.5 * descriptors['mol_weight'] - 50  # Simple approximation
            
            # Normalize key values to prevent large scale differences
            descriptors['normalized_weight'] = np.log1p(descriptors['mol_weight']) / 6.0
            descriptors['normalized_tpsa'] = descriptors['tpsa'] / 150.0  # Typical max ~150
            descriptors['normalized_logP'] = (descriptors['logP'] + 3) / 10.0  # Typically -3 to +7
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return None
    
    @staticmethod
    def get_descriptor_vector(descriptors):
        """
        Convert descriptor dictionary to a torch tensor with improved normalization
        """
        if descriptors is None:
            return None
            
        # Select key descriptors and apply consistent normalization
        keys_and_norms = [
            ('logP', lambda x: (x + 3) / 8),  # LogP typically -3 to +5
            ('tpsa', lambda x: x / 150),      # TPSA typically 0 to 150
            ('h_donors', lambda x: x / 6),    # H-donors typically 0 to 6
            ('h_acceptors', lambda x: x / 10), # H-acceptors typically 0 to 10
            ('mol_weight', lambda x: min(x / 500, 1.0)),  # MW scale with cap at 500
            ('rotatable_bonds', lambda x: min(x / 10, 1.0)),  # Rotatable bonds 0-10
            ('aromatic_rings', lambda x: min(x / 5, 1.0)),   # Aromatic rings 0-5
            ('aliphatic_rings', lambda x: min(x / 3, 1.0))  # Aliphatic rings 0-3
        ]
        
        # Apply normalizations
        values = []
        for key, norm_func in keys_and_norms:
            if key in descriptors:
                values.append(norm_func(descriptors[key]))
            else:
                values.append(0.0)  # Default if missing
        
        return torch.tensor(values, dtype=torch.float).view(1, -1)
