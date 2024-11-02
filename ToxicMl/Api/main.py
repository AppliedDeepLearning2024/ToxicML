import os
import sys
# insert root directory into python module search path
sys.path.insert(1, os.getcwd())



from fastapi import FastAPI
from ToxicMl.MLmodels.base import Endpoint, MlModel, Prediction
from ToxicMl.MLmodels.randomModel import RandomModel
randomModel = RandomModel()


app = FastAPI()



@app.get("/")
async def root():
	return {"message": "Hello world"}


@app.get("/mlmodels")
async def getMlModels() -> list[MlModel]:
	return [
		MlModel(name="RandomModel", endpoint=[Endpoint.endpoint1, Endpoint.endpoint2])
	]

@app.post("/prediction/randommodel/")
async def getPredictionsRandomModel(smiles: list[str]) -> list[Prediction]:
	return randomModel.predictBatch(smiles)
	
