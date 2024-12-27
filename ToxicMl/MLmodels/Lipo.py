from ToxicMl.MLmodels.gcn import ChemGCNReg
from ToxicMl.MLmodels.base import BaseModel

from pathlib import Path
import torch
from torch_geometric.data import Data

class LipoGCNModel(BaseModel):
    def __init__(self, path:Path):
        super().__init__()
        self.name = "LipoGCN5-128"
        self.model = ChemGCNReg(5, 133, 128, 1)
        if path.is_file():
            state_dict = torch.load(path, weights_only=False)
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)
            print("weights loaded")
        
    def predict(self, smile:str):
        graph = self.preprocessor.smiles2graph(smile)
        data = Data(
                    x = torch.tensor(graph["node_feat"], dtype=torch.float32),
                    edge_index = torch.tensor(graph["edge_index"]),
                    y = torch.tensor([]),
                    descriptors = torch.tensor(graph["descriptors"], dtype=torch.float32)
                )
        return self.model(data).item()