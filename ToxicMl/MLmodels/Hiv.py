from pathlib import Path
import torch
from torch_geometric.data import Data
from ToxicMl.MLmodels.attention import ChemAttention

from ToxicMl.MLmodels.base import BaseModel
model = ChemAttention(5, 133, 64, 2)

class HivGNNModel(BaseModel):
    def __init__(self, path:Path):
        super().__init__()
        self.name = "ChemAttention5-125"
        self.model = ChemAttention(5, 133, 128, 2)
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
        res = self.model(data)[:,0]
        return res.item()