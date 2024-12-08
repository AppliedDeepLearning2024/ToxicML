import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from math import isclose

from ToxicMl.metrics import Precision, Recall, F1, Accuracy
from ToxicMl.metrics import MSE, MAE, MaxError

def test_accuracy():
    metric = Accuracy()
    assert metric.compute() == 0

    prediction1 = Uniform(0,1).sample((1000,1))
    target1 = torch.round(Uniform(0,1).sample((1000,1)))

    metric.update(prediction1, target1)
    assert accuracy_score(torch.round(prediction1), target1) == metric.compute()

    prediction2 = Uniform(0,1).sample((1000,1))
    target2 = torch.round(Uniform(0,1).sample((1000,1)))
    metric.update(prediction2, target2)
    predictions = torch.round(torch.tensor(prediction1.tolist() + prediction2.tolist()))
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = accuracy_score(predictions, targets)

    assert sklearn_metric == metric.compute()

    assert "Accuracy" in metric.to_dict().keys()

def test_precission():
    metric = Precision()
    assert metric.compute() == 0

    prediction1 = Uniform(0,1).sample((1000,1))
    target1 = torch.round(Uniform(0,1).sample((1000,1)))


    metric.update(prediction1, target1)
    assert precision_score(torch.round(prediction1), target1) == metric.compute()

    prediction2 = Uniform(0,1).sample((1000,1))
    target2 = torch.round(Uniform(0,1).sample((1000,1)))
    metric.update(prediction2, target2)
    predictions = torch.round(torch.tensor(prediction1.tolist() + prediction2.tolist()))
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = precision_score(predictions, targets)

    assert sklearn_metric == metric.compute()

    assert "Precision" in metric.to_dict().keys()


def test_recall():
    metric = Recall()
    assert metric.compute() == 0

    prediction1 = Uniform(0,1).sample((1000,1))
    target1 = torch.round(Uniform(0,1).sample((1000,1)))


    metric.update(prediction1, target1)
    assert recall_score(torch.round(prediction1), target1) == metric.compute()

    prediction2 = Uniform(0,1).sample((1000,1))
    target2 = torch.round(Uniform(0,1).sample((1000,1)))
    metric.update(prediction2, target2)
    predictions = torch.round(torch.tensor(prediction1.tolist() + prediction2.tolist()))
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = recall_score(predictions, targets)

    assert sklearn_metric == metric.compute()

    assert "Recall" in metric.to_dict().keys()


def test_recall():
    metric = Recall()
    assert metric.compute() == 0

    prediction1 = Uniform(0,1).sample((1000,1))
    target1 = torch.round(Uniform(0,1).sample((1000,1)))


    metric.update(prediction1, target1)
    assert recall_score(torch.round(prediction1), target1) == metric.compute()

    prediction2 = Uniform(0,1).sample((1000,1))
    target2 = torch.round(Uniform(0,1).sample((1000,1)))
    metric.update(prediction2, target2)
    predictions = torch.round(torch.tensor(prediction1.tolist() + prediction2.tolist()))
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = recall_score(predictions, targets)

    assert sklearn_metric == metric.compute()

    assert "Recall" in metric.to_dict().keys()


def test_f1():
    metric = F1()
    assert metric.compute() == 0

    prediction1 = Uniform(0,1).sample((1000,1))
    target1 = torch.round(Uniform(0,1).sample((1000,1)))


    metric.update(prediction1, target1)
    assert isclose(f1_score(torch.round(prediction1), target1), metric.compute())

    prediction2 = Uniform(0,1).sample((1000,1))
    target2 = torch.round(Uniform(0,1).sample((1000,1)))
    metric.update(prediction2, target2)
    predictions = torch.round(torch.tensor(prediction1.tolist() + prediction2.tolist()))
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = f1_score(predictions, targets)

    assert isclose(sklearn_metric,metric.compute())

    assert "F1" in metric.to_dict().keys()


def test_mse():
    metric = MSE()
    assert metric.compute() == 0

    prediction1 = Normal(0,1).sample((1000,1))
    target1 = Normal(0,1).sample((1000,1))


    metric.update(prediction1, target1)
    assert isclose(mean_squared_error(prediction1, target1), metric.compute(), abs_tol=0.001)

    prediction2 = Normal(0,1).sample((1000,1))
    target2 = Normal(0,1).sample((1000,1))
    metric.update(prediction2, target2)
    predictions = torch.tensor(prediction1.tolist() + prediction2.tolist())
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = mean_squared_error(predictions, targets)

    assert isclose(sklearn_metric,metric.compute(), abs_tol=0.001)

    assert "MSE" in metric.to_dict().keys()

def test_mae():
    metric = MAE()
    assert metric.compute() == 0

    prediction1 = Normal(0,1).sample((1000,1))
    target1 = Normal(0,1).sample((1000,1))


    metric.update(prediction1, target1)
    assert isclose(mean_absolute_error(prediction1, target1), metric.compute(), abs_tol=0.001)

    prediction2 = Normal(0,1).sample((1000,1))
    target2 = Normal(0,1).sample((1000,1))
    metric.update(prediction2, target2)
    predictions = torch.tensor(prediction1.tolist() + prediction2.tolist())
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = mean_absolute_error(predictions, targets)

    assert isclose(sklearn_metric,metric.compute(), abs_tol=0.001)

    assert "MAE" in metric.to_dict().keys()

def test_max_error():
    metric = MaxError()
    assert metric.compute() == 0

    prediction1 = Normal(0,1).sample((1000,1))
    target1 = Normal(0,1).sample((1000,1))


    metric.update(prediction1, target1)
    assert isclose(max_error(prediction1, target1), metric.compute(), abs_tol=0.001)

    prediction2 = Normal(0,1).sample((1000,1))
    target2 = Normal(0,1).sample((1000,1))
    metric.update(prediction2, target2)
    predictions = torch.tensor(prediction1.tolist() + prediction2.tolist())
    targets = torch.tensor(target1.tolist() + target2.tolist())
    sklearn_metric = max_error(predictions, targets)

    assert isclose(sklearn_metric,metric.compute(), abs_tol=0.001)

    assert "Max Error" in metric.to_dict().keys()