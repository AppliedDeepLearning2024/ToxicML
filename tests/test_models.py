import pytest
from ToxicMl.MLmodels.randomModel import RandomModel
from ToxicMl.MLmodels.base import Prediction

SMILES = [
    'C1CCC1OCC',
    'CC(C)OCC',
    'CCOCC',
]

models = [
    RandomModel()
]

@pytest.mark.parametrize("model", models)
def test_predict(model):
    res = model.predict(SMILES[0])
    assert isinstance(res, Prediction)
    assert len(res.endpointPredictions) > 1
    assert res.smile == SMILES[0]

@pytest.mark.parametrize("model", models)
def test_predict_error(model):
    with pytest.raises(Exception):
        model.predict("A")


@pytest.mark.parametrize("model", models)
def test_predict_batch(model):
    res = model.predictBatch(SMILES)
    assert len(res) == len(SMILES)
    for i, el in enumerate(res):
        assert isinstance(el, Prediction)
        assert len(el.endpointPredictions) > 1
        assert el.smile == SMILES[i]

@pytest.mark.parametrize("model", models)
def test_predict_batch_error(model):
    with pytest.raises(Exception):
        model.predictBatch(["A", "B", "C"])


 