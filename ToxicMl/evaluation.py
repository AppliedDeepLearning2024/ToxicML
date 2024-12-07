import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.data import Data
import torch


class OGBGTransform(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.x = data.x.to(torch.float32)
        data.y = data.y.flatten()
        return data


def get_hiv_data():
    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", transform=OGBGTransform()) 

    split_idx = dataset.get_idx_split() 
    train_loader = dataset[split_idx["train"]]
    valid_loader = dataset[split_idx["valid"]]
    test_loader = dataset[split_idx["test"]]

    return dataset, train_loader, test_loader, valid_loader

def get_lipo_data():
    dataset = PygGraphPropPredDataset(name = "ogbg-mollipo", transform=OGBGTransform()) 

    split_idx = dataset.get_idx_split() 
    train_loader = dataset[split_idx["train"]]
    valid_loader = dataset[split_idx["valid"]]
    test_loader = dataset[split_idx["test"]]

    return dataset, train_loader, test_loader, valid_loader



def evaluate_hiv(metrics, model, loader, dataset_name):
    if metrics == {}:
        metrics = {
            "dataset" : [],
            "f1" : [],
            "recal" : [],
            "precision": [],
            "accuracy" : [],
        }

    y_pred = []
    y = []
    for data in loader:
        y_pred += model(data).argmax(dim=1).tolist()
        y += data.y.tolist()
    metrics["f1"].append(f1_score(y_pred, y))
    metrics["recal"].append(recall_score(y_pred, y))
    metrics["precision"].append(precision_score(y_pred, y))
    metrics["accuracy"].append(accuracy_score(y_pred, y))
    metrics["dataset"].append(dataset_name)
    return metrics


def evaluate_lipo(metrics, model, loader, dataset_name):
    if metrics == {}:
        metrics = {
            "dataset" : [],
            "mae" : [],
            "mse" : [],
            "max error": [],
            "r2 score" : [],
        }

    y_pred = []
    y = []
    for data in loader:
        y_pred += model(data).argmax(dim=1).tolist()
        y += data.y.tolist()
    metrics["mae"].append(mean_absolute_error(y_pred, y))
    metrics["mse"].append(mean_squared_error(y_pred, y))
    metrics["max error"].append(max_error(y_pred, y))
    metrics["r2 score"].append(r2_score(y_pred, y))
    metrics["dataset"].append(dataset_name)
    return metrics