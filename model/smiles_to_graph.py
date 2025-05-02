import torch
from rdkit import Chem
from torch_geometric.data import Data

class SMILESToGraph:
    def __init__(self):
        self.atomic_numbers = {}
        # Map of atomic number to index
        for i, atom_symbol in enumerate(['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']):
            self.atomic_numbers[atom_symbol] = i
            
    def convert(self, smiles):
        """Convert a SMILES string to a graph representation"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # Get node features
            num_atoms = mol.GetNumAtoms()
            features = torch.zeros(num_atoms, len(self.atomic_numbers) + 4)  # Atomic number + additional features
            
            for atom_idx in range(num_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_symbol = atom.GetSymbol()
                
                # One-hot encode the atom type
                if atom_symbol in self.atomic_numbers:
                    features[atom_idx, self.atomic_numbers[atom_symbol]] = 1
                
                # Add additional atom features
                features[atom_idx, len(self.atomic_numbers)] = atom.GetFormalCharge()
                features[atom_idx, len(self.atomic_numbers) + 1] = atom.GetNumRadicalElectrons()
                features[atom_idx, len(self.atomic_numbers) + 2] = atom.GetIsAromatic()
                features[atom_idx, len(self.atomic_numbers) + 3] = atom.GetTotalNumHs()
            
            # Get edge indices and features
            src = []
            dst = []
            edge_attr = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                bond_type = bond.GetBondType()
                if bond_type == Chem.rdchem.BondType.SINGLE:
                    bond_feature = [1, 0, 0, 0]
                elif bond_type == Chem.rdchem.BondType.DOUBLE:
                    bond_feature = [0, 1, 0, 0]
                elif bond_type == Chem.rdchem.BondType.TRIPLE:
                    bond_feature = [0, 0, 1, 0]
                elif bond_type == Chem.rdchem.BondType.AROMATIC:
                    bond_feature = [0, 0, 0, 1]
                else:
                    bond_feature = [0, 0, 0, 0]
                
                # Add edges in both directions
                src.extend([i, j])
                dst.extend([j, i])
                edge_attr.extend([bond_feature, bond_feature])
            
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            if len(edge_attr) > 0:
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                edge_attr = torch.zeros((0, 4), dtype=torch.float)
            
            # Create PyG data object
            data = Data(
                x=features, 
                edge_index=edge_index,
                edge_attr=edge_attr,
                smiles=smiles
            )
            
            return data
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {e}")
            return None
