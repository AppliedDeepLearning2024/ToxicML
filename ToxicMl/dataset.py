from torch_geometric.data import InMemoryDataset
import pandas as pd
from pathlib import Path
import gzip
from ToxicMl.MLmodels.preprocessing import ChemicalPreprocessor
from torch_geometric.data import Data
from tqdm import tqdm
import torch

class HivDataset(InMemoryDataset):
    def __init__(self, root:Path, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.mol_path = root / "mapping" / "mol.csv.gz"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.init_splits()
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        preprocessor = ChemicalPreprocessor()
        data_list = []
        with gzip.open(self.mol_path,'rt') as f:
            next(f)
            for line in tqdm(f):
                y, smiles, _ = line.split(",")
                graph = preprocessor.smiles2graph(smiles)
                data_list.append(
                    Data(
                        x = torch.tensor(graph["node_feat"], dtype=torch.float32),
                        edge_index = torch.tensor(graph["edge_index"]),
                        y = torch.tensor(int(y)),
                        descriptors = torch.tensor(graph["descriptors"], dtype=torch.float32)
                    )
                )
        s = [
            el.descriptors for el in data_list
        ]
        s = torch.cat(s)
        means = s.mean(dim=0)
        std = s.std(dim=0)
        normalized = (s - means) / std
        nan_mask = torch.isnan(normalized).any(dim=0)
        for i, el in enumerate(data_list):
            el.descriptors = normalized[i,~nan_mask]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def init_splits(self):
        self.split_path = self.root / "split/scaffold"
        train_index = pd.read_csv(
            self.split_path/"train.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        test_index = pd.read_csv(
            self.split_path/"test.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        validation_index = pd.read_csv(
            self.split_path/"valid.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        self.splits = {
            "train": train_index,
            "valid": validation_index,
            "test": test_index,
        }
    def get_idx_split(self):
        return self.splits
    

class LipoDataset(InMemoryDataset):
    def __init__(self, root:Path, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.mol_path = root / "mapping" / "mol.csv.gz"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.init_splits()
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        preprocessor = ChemicalPreprocessor()
        data_list = []
        with gzip.open(self.mol_path,'rt') as f:
            next(f)
            for line in tqdm(f):
                y, smiles, _ = line.split(",")
                graph = preprocessor.smiles2graph(smiles)
                data_list.append(
                    Data(
                        x = torch.tensor(graph["node_feat"], dtype=torch.float32),
                        edge_index = torch.tensor(graph["edge_index"]),
                        y = torch.tensor(float(y), dtype=torch.float32),
                        descriptors = torch.tensor(graph["descriptors"], dtype=torch.float32)
                    )
                )
        s = [
            el.descriptors for el in data_list
        ]
        s = torch.cat(s)
        means = s.mean(dim=0)
        std = s.std(dim=0)
        normalized = (s - means) / std
        nan_mask = torch.isnan(normalized).any(dim=0)
        for i, el in enumerate(data_list):
            el.descriptors = normalized[i,~nan_mask]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def init_splits(self):
        self.split_path = self.root / "split/scaffold"
        train_index = pd.read_csv(
            self.split_path/"train.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        test_index = pd.read_csv(
            self.split_path/"test.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        validation_index = pd.read_csv(
            self.split_path/"valid.csv.gz",
            compression="gzip",
            header=None).values.flatten().tolist()
        self.splits = {
            "train": train_index,
            "valid": validation_index,
            "test": test_index,
        }
    def get_idx_split(self):
        return self.splits


