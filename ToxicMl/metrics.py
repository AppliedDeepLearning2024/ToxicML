from abc import ABCMeta, abstractmethod
import torch
from statistics import mean

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def compute(self) -> float:
        '''
        computes the metric and returns it
        '''

        pass


    @abstractmethod
    def __dict__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self) -> None:
        self.n = 0
        self.n_correct = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.n = 0
        self.n_correct = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        prediction = torch.round(prediction).to(torch.int8)
        self.n += len(prediction)
        self.n_correct += (prediction == target).sum().item()
        

    def to_dict(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        accuracy = round(self.compute(),4)

        return {"Accuracy": accuracy}


    def compute(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        return 0.0 if self.n == 0 else  self.n_correct / self.n
    

       

class Precision(PerformanceMeasure):
    '''
    Precission of binary classification problem.
    '''

    def __init__(self) -> None:
        self.tp = 0
        self.p = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.tp = 0
        self.p = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''
        prediction = torch.round(prediction)
        self.p += target.sum().item()
        self.tp += torch.logical_and(prediction, target).sum().item()

    def compute(self) -> float:
        '''
        Compute and return the precision as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        return 0.0 if self.p == 0 else  self.tp / self.p
    
    def to_dict(self):
        return {
            "Precision" : self.compute()
        }
       

class Recall(PerformanceMeasure):
    '''
    Recal of binary classification problem.
    '''

    def __init__(self) -> None:
        self.tp = 0
        self.p = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.tp = 0
        self.p = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        prediction = torch.round(prediction)
        self.p += prediction.sum().item()
        self.tp += torch.logical_and(prediction, target).sum().item()

    def compute(self) -> float:
        '''
        Compute and return the precision as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        return 0.0 if self.p == 0 else  self.tp / self.p
    
    def to_dict(self):
        return {
            "Recall" : self.compute()
        }
    
class F1(PerformanceMeasure):
    '''
    F1 of binary classification problem.
    '''

    def __init__(self) -> None:
        self.tp = 0
        self.relevant = 0
        self.retrived = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.tp = 0
        self.relevant = 0
        self.retrived = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        prediction = torch.round(prediction)
        self.retrived += prediction.sum().item()
        self.relevant += target.sum().item()
        self.tp += torch.logical_and(prediction, target).sum().item()

    def compute(self) -> float:
        '''
        Compute and return the precision as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        precision = 0 if self.retrived == 0 else self.tp / self.retrived
        recall = 0 if self.relevant == 0 else self.tp / self.relevant
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    
    def to_dict(self):
        return {
            "F1" : self.compute()
        }
    
class MSE(PerformanceMeasure):
    '''
    MEAN SQUARED ERROR
    '''

    def __init__(self) -> None:
        self.sum_abs_diff = 0
        self.n = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.sum_sqrd_diff = 0
        self.n = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        self.sum_sqrd_diff += torch.pow(prediction - target, 2).sum().item()
        self.n += len(prediction)

    def compute(self) -> float:
        '''
        Compute and return the mean squared error
        '''
        return 0 if self.n == 0 else self.sum_sqrd_diff / self.n
    
    def to_dict(self):
        return {
            "MSE" : self.compute()
        }
    
class MAE(PerformanceMeasure):
    '''
    MEAN ABSOLUTE ERROR
    '''

    def __init__(self) -> None:
        self.sum_abs_diff = 0
        self.n = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.sum_abs_diff = 0
        self.n = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        self.sum_abs_diff += torch.abs(prediction - target).sum().item()
        self.n += len(prediction)

    def compute(self) -> float:
        '''
        Compute and return the mean squared error
        '''
        return 0 if self.n == 0 else self.sum_abs_diff / self.n
    
    def to_dict(self):
        return {
            "MAE" : self.compute()
        }
    

class MaxError(PerformanceMeasure):
    '''
    MAX ERROR
    '''

    def __init__(self) -> None:
        self.max_error = 0
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.max_error = 0

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        '''

        max_error = torch.abs(prediction - target).max()
        self.max_error = max(max_error, self.max_error)

    def compute(self) -> float:
        '''
        Compute and return the mean squared error
        '''
        return self.max_error
    
    def to_dict(self):
        return {
            "Max Error" : self.compute()
        }