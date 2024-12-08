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


def testGetGraphFromSmile():
    def check_graph(graph):
        assert 'edge_index' in graph.keys()
        assert 'edge_feat' in graph.keys()
        assert 'node_feat' in graph.keys()
        assert 'num_nodes' in graph.keys()
        assert 'descriptors' in graph.keys()

        assert graph["num_nodes"] > 0

    graph = preprocessor.smiles2graph(SMILES[0])
    check_graph(graph)

    graph = preprocessor.smiles2graph(SMILES[1])
    check_graph(graph)

    graph = preprocessor.smiles2graph(SMILES[2])
    check_graph(graph)
