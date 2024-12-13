import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import numpy as np
import wandb
from torch_geometric.data import DataLoader



#from dlvc.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

class GenericClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def _get_run(self, project: str, group: str, run_name:str, **kwargs):
        return wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=kwargs
        )
    
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metrics,
                 val_metrics,
                 train_data,
                 val_data,
                 test_data,
                 device,
                 train_sampler,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,):
        
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.train_data, self.val_data = train_data, val_data
        self.test_data = test_data
        self.train_sampler = train_sampler

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        self.validation_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size)


        self.train_epoch_steps = np.ceil(len(train_data) / batch_size)
        self.val_epoch_steps = np.ceil(len(val_data) / batch_size)

        

        
    def _reset_metrics(self, metrics):
        for metric in metrics:
            metric.reset()

    def _update_metrics(self, metrics, prediction, target):
        for metric in metrics:
            metric.update(prediction, target)

    def _merge_dicts(self, dicts):
        new_dict = {}
        for d in dicts:
            for k, v in d.items():
                new_dict[k] = v
        return new_dict

    def _log_metrics(self):
        train_metrics = [
            metric.to_dict()
            for metric in self.train_metrics
        ]
        train_metrics = self._merge_dicts(train_metrics)
        train_metrics = {
            f"train/{key}": value
            for key, value in train_metrics.items()
        }

        val_metrics = [
            metric.to_dict()
            for metric in self.val_metrics
        ]
        val_metrics = self._merge_dicts(val_metrics)
        val_metrics = {
            f"validation/{key}": value
            for key, value in val_metrics.items()
        }
        return self._merge_dicts([train_metrics, val_metrics])
        
            
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        self._reset_metrics(self.train_metrics)
        self.model.train()
        for batch_idx, data in tqdm(enumerate(self.train_loader),desc="train epoch", total=self.train_epoch_steps):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output,data.y)
            self._update_metrics(self.train_metrics, output[:,1], data.y)
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()  # adjust learning rate every 5 batches
        return loss.item(), self.train_metrics[0].compute()


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.validation_loader), desc="val epoch", total=self.val_epoch_steps, ):
                data = data.to(self.device)
                output = self.model(data)
                self._update_metrics(self.val_metrics,output[:,1], data.y)
                loss = self.loss_fn(output,data.y).item()
        return loss, self.val_metrics[0].compute()

    def _eval_on_test(self):
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                self._update_metrics(self.val_metrics,output[:,1], data.y)
            columns = []
            rows = []
            for metric in self.val_metrics:
                for k, v in metric.to_dict().items():
                    columns.append(k)
                    rows.append(v)
            table = wandb.Table(columns=columns, data=[rows])
            self.run.log({"HIV Test results": table})

    def train(self, run_name:str) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.run = self._get_run(
            "ToxicML",
            "DeepLearningHiv",
            run_name,
            loss_function = type(self.loss_fn).__name__,
            optimizer = type(self.optimizer).__name__,
            num_epochs = self.num_epochs,
            batch_size = self.batch_size,
            sampler = True if self.train_sampler != None else False,
            trainable_params=params
        )
        best_weights = None
        best_score = -np.inf
        try:
            for i in range(0, self.num_epochs):
                train_loss, _ = self._train_epoch(i)
                val_loss, score = self._val_epoch(i)
                metrics = self._log_metrics()
                metrics["train/loss"] = train_loss
                metrics["validation/loss"] = val_loss
                self.run.log(metrics)

                if score > best_score:
                    best_score = score
                    best_weights = self.model.state_dict()
            self.model.load_state_dict(best_weights)
            if self.test_data is not None:
                self._eval_on_test()
            
        finally:
            self.run.finish()



class GenericRegressionTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def _get_run(self, project: str, group: str, run_name:str, **kwargs):
        return wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=kwargs
        )
    
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metrics,
                 val_metrics,
                 train_data,
                 val_data,
                 test_data,
                 device,
                 train_sampler,
                 num_epochs: int, 
                 training_save_dir: Path,
                 batch_size: int = 4,):
        
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch 
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th 
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        '''
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.train_data, self.val_data = train_data, val_data
        self.test_data = test_data
        self.train_sampler = train_sampler

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        self.validation_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size)


        self.train_epoch_steps = np.ceil(len(train_data) / batch_size)
        self.val_epoch_steps = np.ceil(len(val_data) / batch_size)

        

        
    def _reset_metrics(self, metrics):
        for metric in metrics:
            metric.reset()

    def _update_metrics(self, metrics, prediction, target):
        for metric in metrics:
            metric.update(prediction, target)

    def _merge_dicts(self, dicts):
        new_dict = {}
        for d in dicts:
            for k, v in d.items():
                new_dict[k] = v
        return new_dict

    def _log_metrics(self):
        train_metrics = [
            metric.to_dict()
            for metric in self.train_metrics
        ]
        train_metrics = self._merge_dicts(train_metrics)
        train_metrics = {
            f"train/{key}": value
            for key, value in train_metrics.items()
        }

        val_metrics = [
            metric.to_dict()
            for metric in self.val_metrics
        ]
        val_metrics = self._merge_dicts(val_metrics)
        val_metrics = {
            f"validation/{key}": value
            for key, value in val_metrics.items()
        }
        all_metrics = self._merge_dicts([train_metrics, val_metrics])
        return all_metrics

            
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
        self._reset_metrics(self.train_metrics)
        self.model.train()
        for batch_idx, data in tqdm(enumerate(self.train_loader),desc="train epoch", total=self.train_epoch_steps):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output,data.y)
            self._update_metrics(self.train_metrics, output[:,-1], data.y)
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()  # adjust learning rate every 5 batches
        return loss.item(), self.train_metrics[0].compute()


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 

        epoch_idx (int): Current epoch number
        """
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(self.validation_loader), desc="val epoch", total=self.val_epoch_steps, ):
                data = data.to(self.device)
                output = self.model(data)
                self._update_metrics(self.val_metrics,output[:,-1], data.y)
                loss = self.loss_fn(output,data.y).item()
        return loss, self.val_metrics[0].compute()

    def _eval_on_test(self):
        self.model.eval()
        self._reset_metrics(self.val_metrics)
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                output = self.model(data)
                self._update_metrics(self.val_metrics,output[:,-1], data.y)
            columns = []
            rows = []
            for metric in self.val_metrics:
                for k, v in metric.to_dict().items():
                    columns.append(k)
                    rows.append(v)
            table = wandb.Table(columns=columns, data=[rows])
            self.run.log({"test/LIPO Test results": table})

    def _log_residuals_scatterplot(self, dataLoader: DataLoader, table_name):
        prediction = []
        target = []
        for data in dataLoader:
            output = self.model(data)
            prediction += output.flatten().tolist()
            target += data.y.flatten().tolist()
        data = [[x, y] for (x, y) in zip(target, prediction)]
        table = wandb.Table(data=data, columns = ["target", "prediction"])
        self.run.log({table_name : wandb.plot.scatter(table, "target", "prediction",title="Prediction vs Target Residuals")})



    def train(self, run_name:str) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.run = self._get_run(
            "ToxicML",
            "DeepLearningLipo",
            run_name,
            loss_function = type(self.loss_fn).__name__,
            optimizer = type(self.optimizer).__name__,
            num_epochs = self.num_epochs,
            batch_size = self.batch_size,
            sampler = True if self.train_sampler != None else False,
            trainable_params = params
        )
        
        best_score = np.inf
        best_weights = None
        try:
            for i in range(0, self.num_epochs):
                train_loss, _ = self._train_epoch(i)
                val_loss, score = self._val_epoch(i)
                metrics_dict = self._log_metrics()
                metrics_dict["train/loss"] = train_loss
                metrics_dict["validation/loss"] = val_loss
                self.run.log(metrics_dict)
                if score < best_score:
                    best_score = score
                    best_weights = self.model.state_dict()
            
            self.model.load_state_dict(best_weights)
            self._log_residuals_scatterplot(self.train_loader, "train/residuals")
            self._log_residuals_scatterplot(self.validation_loader, "validation/residuals")
            if self.test_data is not None:
                self._eval_on_test()
                self._log_residuals_scatterplot(self.test_loader, "test/residuals")
            
        finally:
            self.run.finish()


            

                





            
            


