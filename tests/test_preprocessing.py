from ToxicMl.MLmodels.preprocessing import ChemicalPreprocessor
from rdkit.Chem import Mol
import numpy as np

SMILES = [
    'C1CCC1OCC',
    'CC(C)OCC',
    'CCOCC',
]

preprocessor = ChemicalPreprocessor()


def testGetMoleculesFromSMILES():
    molecules = preprocessor.getMoleculesFromSmiles(SMILES)
    assert len(molecules) == len(SMILES)
    for el in molecules:
        assert isinstance(el, Mol) 

def testGetMoleculesFromSMILESIncorrectSmiles():
    SMILES = ["A", "B", "C"]
    try:
        molecules = preprocessor.getMoleculesFromSmiles(SMILES)
        assert len(molecules) == len(SMILES)
        assert False, "Error should be raised"
    except:
        assert True   

def testGetFingerprintFromMol():
    molecules = preprocessor.getMoleculesFromSmiles(SMILES)
    prints = preprocessor.getFingerprintFromMol(molecules)
    isinstance(prints, np.ndarray)
    assert prints.shape == (3, 2024)

def testGetDescriptorsFromMol():
    molecules = preprocessor.getMoleculesFromSmiles(SMILES)
    descriptors = preprocessor.getDescriptorsFromMol(molecules)
    isinstance(descriptors, np.ndarray)
    assert descriptors.shape == (3, 210) 
    print(descriptors.shape)