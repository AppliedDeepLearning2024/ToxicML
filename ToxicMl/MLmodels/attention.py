import torch
import torch.nn.functional as F

from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.nn.conv import GATv2Conv

class ChemAttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChemAttentionBlock, self).__init__()
        self.conv = GATv2Conv(in_channels, out_channels, 1)
        self.relu = ReLU(inplace=True)
        self.norm = BatchNorm(out_channels)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        x = self.relu(x)
        x = F.dropout(x)
        x = self.norm(x)
        data.x = x
        return data


class ChemAttention(torch.nn.Module):
    def __init__(self, embedder_depth, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = torch.nn.ModuleList([
            ChemAttentionBlock(in_channels=in_channels, out_channels=hidden_channels)
            if i == 0
            else ChemAttentionBlock(in_channels=hidden_channels, out_channels=hidden_channels)
            for i in range(embedder_depth)])
        self.linear1 = Linear(hidden_channels, 8)
        self.relu1 = ReLU(inplace=True)
        self.linear2 = Linear(8, out_channels)
        self.relu2 = ReLU(inplace=True)
        

    def forward(self, data: Data):
        for module in self.encoder:
            data = module(data)
        x = global_max_pool(data.x, data.batch)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = F.softmax(x, dim=1)
        return x
    
class ChemAttentionReg(torch.nn.Module):
    def __init__(self, embedder_depth, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = torch.nn.ModuleList([
            ChemAttentionBlock(in_channels=in_channels, out_channels=hidden_channels)
            if i == 0
            else ChemAttentionBlock(in_channels=hidden_channels, out_channels=hidden_channels)
            for i in range(embedder_depth)])
        self.linear1 = Linear(hidden_channels, 8)
        self.relu1 = ReLU(inplace=True)
        self.linear2 = Linear(8, out_channels)
        

    def forward(self, data: Data):
        for module in self.encoder:
            data = module(data)
        x = global_max_pool(data.x, data.batch)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x