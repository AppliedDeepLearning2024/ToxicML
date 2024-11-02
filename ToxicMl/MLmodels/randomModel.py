import numpy as np
from pathlib import Path
import random as rnd

from ToxicMl.MLmodels.base import BaseModel, Prediction, EndpointPrediction, Endpoint
    

class RandomModel(BaseModel):
    def __init__(self, seed = None, name="UniformRandomModel"):
        self.name = name
        self.seed = seed
        self.generator = rnd.Random()
        self.generator.seed(seed)

    def predict(self, smile: str) -> Prediction:
        predictions = [   
            EndpointPrediction(endpoint=endpoint, value=np.random.uniform(0,1))
            for endpoint in Endpoint
        ]
        return Prediction(smile=smile, endpointPredictions=predictions)
    
    def predictBatch(self, smiles: list[str]) -> list[Prediction]:
        return [
            self.predict(smile)
            for smile in smiles
        ]

    def save(self, filepath: Path):
        pass
    
    def load(self, filepath: Path):
        pass
    
    def __str__(self, ):
        return f"{self.name=},{self.seed=}"
    