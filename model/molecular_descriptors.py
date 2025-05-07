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
        Convert descriptor dictionary to a torch tensor for model input
        
        Args:
            descriptors: Dictionary of molecular descriptors
            
        Returns:
            descriptor_vector: Torch tensor with descriptor values
        """
        if descriptors is None:
            return None
            
        # Select key descriptors most relevant for solubility
        # These are approximately in order of importance for solubility prediction
        keys = [
            'normalized_logP',     # Most important for solubility
            'normalized_tpsa',     # Polarity greatly affects solubility
            'h_donors', 
            'h_acceptors', 
            'normalized_weight',
            'rotatable_bonds',
            'aromatic_rings',
            'aliphatic_rings'
        ]
        
        values = [descriptors[k] for k in keys]
        return torch.tensor(values, dtype=torch.float).view(1, -1)
