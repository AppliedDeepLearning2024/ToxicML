from molvs import standardize_smiles
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Mol
from rdkit.Chem import Descriptors
from rdkit import Chem
from pandas import DataFrame
import numpy as np
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from torch_geometric.data import Data


allowable_features = {
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


class ChemicalPreprocessor:
    def __init__(
            self,
            morgan_radius = 3,
            morgan_size = 2024):
        """
        morgan_radius -> the radius used to compute morgan fingerprints
        morgan_size -> the size of the morgan fingerprint vectors
        """
        self.morgan_radius = morgan_radius
        self.morgan_size = morgan_size
        self.fpgen = AllChem.GetMorganGenerator(radius=morgan_radius,fpSize=morgan_size)
        self.atom_encoder = MultiHotAtomFeaturizer.v1()

    def ReorderCanonicalRankAtoms(mol):
        order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
        mol_renum = Chem.RenumberAtoms(mol, order)
        return mol_renum, order

    def getMoleculesFromSmiles(self, codes: list[str]) -> list[Mol]:
        chemicals = [
            MolFromSmiles(standardize_smiles(el))
            for el in codes
        ]
        return chemicals

    def getFingerprintFromMol(self, mols: list[Mol]) -> np.ndarray:
        fingerprints = [
            self.fpgen.GetFingerprintAsNumPy(mol)
            for mol in mols
        ]

        return np.stack(fingerprints, axis=0)
    
    def getDescriptorsFromMol(self, mols: list[Mol]) -> np.ndarray:
        descriptors = DataFrame([
            Descriptors.CalcMolDescriptors(mol)
            for mol in mols
        ])
        return descriptors.values
    
    def bond_to_feature_vector(self, bond):
        """
        Converts rdkit bond object to feature list of indices
        :param mol: rdkit bond object
        :return: list
        """
        bond_feature = [
                    safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                    allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                    allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                ]
        return bond_feature


    def smiles2graph(self, smiles_string, removeHs=True):
        """
        Converts SMILES string to graph Data object
        :input: SMILES string (str)
        :return: graph object
        """

        mol = Chem.MolFromSmiles(smiles_string)
        mol = mol if removeHs else Chem.AddHs(mol)

        # atoms
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_encoder(atom))
        x = np.array(atom_features_list, dtype = np.int64)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = self.bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        graph['descriptors'] = np.nan_to_num(self.getDescriptorsFromMol([mol]), copy=True)

        return graph 