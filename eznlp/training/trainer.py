# -*- coding: utf-8 -*-
from typing import List
import time
import numpy
import logging
import torch

from ..data.wrapper import Batch
from ..data.dataset import Dataset

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    References
    ----------
    [1] https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 num_metrics: int=1, 
                 optimizer: torch.optim.Optimizer=None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler=None, 
                 schedule_by_step: bool=False, 
                 device: torch.device=None, 
                 grad_clip: float=None, 
                 use_amp: bool=False):
        self.model = model
        self.num_metrics = num_metrics
        
        self.optimizer = optimizer
        if schedule_by_step:
            assert not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.scheduler = scheduler
        self.schedule_by_step = schedule_by_step
        
        assert device is not None
        self.device = device
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        
    def forward_batch(self, batch: Batch):
        """
        Forward to the loss (scalar). 
        Optionally return the gold and predicted labels of the batch for evaluation. 
        
        Returns
        -------
        A Tuple of (loss, ) or (loss, (y_gold_1, y_pred_1), (y_gold_2, y_pred_2), ...)
        """
        raise NotImplementedError("Not Implemented `forward_batch`")
        
        
    def backward_batch(self, loss: torch.Tensor):
        """
        Backward propagation and update the weights. 
        
        Parameters
        ----------
        loss: torch.Tensor
            A scalar tensor. 
        """
        # Backward propagation
        self.scaler.scale(loss).backward()
        
        if self.grad_clip is not None and self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
        # Update weights
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.scheduler is not None and self.schedule_by_step:
            self.scheduler.step()
            
            
    def evaluate(self, *set_y: List[List[list]]):
        """
        Calculate the metric (i.e., accuracy or F1) evaluating the predicted results against the gold results. 
        This method should typically evaluate over a full dataset, although it also compatibly evaluates over a batch. 
        
        Parameters
        ----------
        set_y: List[List[list]]
            In shape of `num_metrics` * 2 * `num_examples`, like [[y_gold_1, y_pred_1], [y_gold_2, y_pred_2], ...]
            
        Returns
        -------
        metric: Tuple[float]
        """
        raise NotImplementedError("Not Implemented `evaluate`")
        
        
    def predict(self, dataset: Dataset, batch_size: int=32):
        raise NotImplementedError("Not Implemented `predict`")
        
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
        Notes
        -----
        If `schedule_by_step` is False, `scheduler.step` should be properly called after this method. 
        """
        self.model.train()
        
        epoch_losses = []
        epoch_y = [[[], []] for k in range(self.num_metrics)]
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, *possible_batch_y = self.forward_batch(batch)
            self.backward_batch(loss)
            
            epoch_losses.append(loss.item())
            if possible_batch_y:
                for k in range(self.num_metrics):
                    epoch_y[k][0].extend(possible_batch_y[k][0])
                    epoch_y[k][1].extend(possible_batch_y[k][1])
        
        if epoch_y:
            return (numpy.mean(epoch_losses), *self.evaluate(*epoch_y))
        else:
            return (numpy.mean(epoch_losses), )
        
        
    def eval_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        
        epoch_losses = []
        epoch_y = [[[], []] for k in range(self.num_metrics)]
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss, *possible_batch_y = self.forward_batch(batch)
                
                epoch_losses.append(loss.item())
                if possible_batch_y:
                    for k in range(self.num_metrics):
                        epoch_y[k][0].extend(possible_batch_y[k][0])
                        epoch_y[k][1].extend(possible_batch_y[k][1])
        
        if epoch_y:
            return (numpy.mean(epoch_losses), *self.evaluate(*epoch_y))
        else:
            return (numpy.mean(epoch_losses), )
    
    
    def train_steps(self, 
                    train_loader: torch.utils.data.DataLoader, 
                    dev_loader: torch.utils.data.DataLoader=None, 
                    num_epochs: int=10, 
                    max_steps: int=None, 
                    disp_every_steps: int=None, 
                    eval_every_steps: int=None, 
                    save_callback=None, 
                    save_by_loss: bool=True, 
                    save_every_steps: int=None):
        
        max_steps = numpy.inf if max_steps is None else max_steps
        disp_every_steps = len(train_loader) if disp_every_steps is None else disp_every_steps
        eval_every_steps = len(train_loader) if eval_every_steps is None else eval_every_steps
        save_every_steps = numpy.inf if save_every_steps is None else save_every_steps
        if eval_every_steps % disp_every_steps != 0:
            raise ValueError(f"`eval_every_steps` {eval_every_steps} should be multiples of `disp_every_steps` {disp_every_steps}")
            
        self.model.train()
        
        best_dev_loss = numpy.inf
        # The `metric` must hold that it is better if higher, e.g., accuracy or F1. 
        best_dev_metric = -numpy.inf
        
        train_losses = []
        train_y = [[[], []] for k in range(self.num_metrics)]
        eidx, sidx = 0, 0
        done_training = False
        t0 = time.time()
        
        while eidx < num_epochs:
            for batch in train_loader:
                batch = batch.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, *possible_batch_y = self.forward_batch(batch)
                self.backward_batch(loss)
                
                train_losses.append(loss.item())
                if possible_batch_y:
                    for k in range(self.num_metrics):
                        train_y[k][0].extend(possible_batch_y[k][0])
                        train_y[k][1].extend(possible_batch_y[k][1])
                
                if (sidx+1) % disp_every_steps == 0:
                    elapsed_secs = int(time.time() - t0)
                    lrs = [group['lr'] for group in self.optimizer.param_groups]
                    disp_running_info(eidx=eidx, sidx=sidx, lrs=lrs, 
                                      elapsed_secs=elapsed_secs, 
                                      loss=numpy.mean(train_losses),
                                      metric=self.evaluate(*train_y) if train_y else None,
                                      partition='train')
                    train_losses = []
                    train_y = [[[], []] for k in range(self.num_metrics)]
                    t0 = time.time()
                
                if (sidx+1) % eval_every_steps == 0 and dev_loader is not None:
                    dev_loss, *possible_dev_metric = self.eval_epoch(dev_loader)
                    possible_dev_metric = possible_dev_metric if len(possible_dev_metric) > 0 else None
                    
                    elapsed_secs = int(time.time() - t0)
                    disp_running_info(elapsed_secs=elapsed_secs, 
                                      loss=dev_loss, 
                                      metric=possible_dev_metric, 
                                      partition='dev')
                    
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        if (save_callback is not None) and save_by_loss:
                            save_callback(self.model)
                        
                    if (possible_dev_metric is not None) and numpy.mean(possible_dev_metric) > best_dev_metric:
                        best_dev_metric = numpy.mean(possible_dev_metric)
                        if (save_callback is not None) and (not save_by_loss):
                            save_callback(self.model)
                    
                    if self.scheduler is not None and not self.schedule_by_step:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if save_by_loss:
                                assert self.scheduler.mode == 'min'
                                self.scheduler.step(dev_loss)
                            else:
                                assert self.scheduler.mode == 'max'
                                self.scheduler.step(numpy.mean(possible_dev_metric))
                        else:
                            self.scheduler.step()
                            
                    self.model.train()
                    t0 = time.time()
                    
                if (sidx+1) % save_every_steps == 0:
                    if save_callback is not None:
                        save_callback(self.model)
                    
                if (sidx+1) >= max_steps:
                    done_training = True
                    break
                sidx += 1
            
            if done_training:
                break
            eidx += 1




def disp_running_info(eidx=None, sidx=None, lrs=None, elapsed_secs=None, loss=None, metric=None, partition='train'):
    disp_text = []
    if eidx is not None:
        disp_text.append(f"Epoch: {eidx+1}")
    if sidx is not None:
        disp_text.append(f"Step: {sidx+1}")
    if lrs is not None:
        disp_text.append("LR: (" + "/".join(f"{lr:.6f}" for lr in lrs) + ")")
    if len(disp_text) > 0:
        logger.info(" | ".join(disp_text))
        
    if partition.lower().startswith('train'):
        partition = "Train"
    elif partition.lower().startswith('dev'):
        partition = "Dev. "
    elif partition.lower().startswith('test'):
        partition = "Test "
    else:
        raise ValueError("Invalid partition {partition}")
        
    disp_text = []
    assert loss is not None
    disp_text.append(f"\t{partition} Loss: {loss:.3f}")
    if metric is not None:
        disp_text.append(f"{partition} Metrics: " + "/".join(f"{m*100:.2f}%" for m in metric))
    else:
        disp_text.append(f"{partition} PPL: {numpy.exp(loss):.3f}")
    if elapsed_secs is not None:
        mins, secs = elapsed_secs // 60, elapsed_secs % 60
        disp_text.append(f"Elapsed Time: {mins}m {secs}s")
    logger.info(" | ".join(disp_text))
    
    