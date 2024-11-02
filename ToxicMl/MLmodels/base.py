import numpy as np
from pathlib import Path
from pydantic import BaseModel
from enum import Enum


from ToxicMl.MLmodels.preprocessing import ChemicalPreprocessor


class Endpoint(str, Enum):
    endpoint1 = "endpoint1"
    endpoint2 = "endpoint2"

class MlModel(BaseModel):
    name: str
    endpoint: list[Endpoint]

class EndpointPrediction(BaseModel):
    endpoint: Endpoint
    value: float

class Prediction(BaseModel):
    smile: str
    endpointPredictions: list[EndpointPrediction]

class BaseModel:
    def __init__(self):
        self.preprocessor = ChemicalPreprocessor()

    def checkInput(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Incorrect type of X. Is of type {type(X)} not ndarray")
        ndim = len(X.shape)
        if ndim != 2:
            raise ValueError(f"Incorrect number of dimensions. Expected 2 got {ndim}")
        n, m = X.shape
        if n != 1:
            raise ValueError(f"incorrect shape: {X.shape}")
        
    def checkBatchInput(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Incorrect type of X. Is of type {type(X)} not ndarray")
        ndim = len(X.shape)
        if ndim != 2:
            raise ValueError(f"Incorrect number of dimensions. Expected 2 got {ndim}")

    def predict(self, smiles: str) -> Prediction:
        raise NotImplementedError()
    
    def predictBatch(self, smiles: list[str]) -> list[Prediction]:
        raise NotImplementedError()

    def save(self, filepath: Path):
        raise NotImplementedError()
    
    def load(self, filepath: Path):
        raise NotImplementedError()
    
    def __str__(self, ):
        raise NotImplementedError()
    

