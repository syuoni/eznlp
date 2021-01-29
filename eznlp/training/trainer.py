# -*- coding: utf-8 -*-
import time
import numpy as np
import torch

from ..data import Batch


class Trainer(object):
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer=None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler=None, 
                 device=None, 
                 grad_clip: float=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        
        assert device is not None
        
        
    def forward_batch(self, batch: Batch):
        """
        Forward to the loss (scalar). 
        Optionally return the gold and predicted labels of the batch, for evaluation. 
        
        Returns
        -------
        A Tuple of (loss, ) or (loss, y_gold, y_pred)
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
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.grad_clip is not None:
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update weights
        self.optimizer.step()
        
        
    def evaluate(self, y_gold: list, y_pred: list):
        """
        Calculate the metric (i.e., accuracy or F1) evaluating the predicted results 
        against the gold results. It typically evaluate over the full dataset, 
        or compatibly over a batch. 
        """
        raise NotImplementedError("Not Implemented `evaluate`")
        
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.train()
        
        epoch_losses = []
        epoch_y_gold, epoch_y_pred = [], []
        for batch in dataloader:
            batch = batch.to(self.device)
            loss, *possible_batch_y = self.forward_batch(batch)
            self.backward_batch(loss)
            
            epoch_losses.append(loss.item())
            if possible_batch_y:
                epoch_y_gold.extend(possible_batch_y[0])
                epoch_y_pred.extend(possible_batch_y[1])
            
        if epoch_y_gold:
            return (np.mean(epoch_losses), self.evaluate(epoch_y_gold, epoch_y_pred))
        else:
            return (np.mean(epoch_losses), )
        
        
    def eval_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        
        epoch_losses = []
        epoch_y_gold, epoch_y_pred = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                loss, *possible_batch_y = self.forward_batch(batch)
                
                epoch_losses.append(loss.item())
                if possible_batch_y:
                    epoch_y_gold.extend(possible_batch_y[0])
                    epoch_y_pred.extend(possible_batch_y[1])
            
        if epoch_y_gold:
            return (np.mean(epoch_losses), self.evaluate(epoch_y_gold, epoch_y_pred))
        else:
            return (np.mean(epoch_losses), )
    
    
    def train_steps(self, 
                    train_loader: torch.utils.data.DataLoader, 
                    eval_loader: torch.utils.data.DataLoader=None, 
                    n_epochs: int=10, 
                    max_steps: int=None, 
                    disp_every_steps: int=None, 
                    eval_every_steps: int=None, 
                    save_callback=None, 
                    save_by_loss: bool=True, 
                    save_every_steps: int=None, 
                    verbose: bool=True):
        
        max_steps = np.inf if max_steps is None else max_steps
        disp_every_steps = len(train_loader) if disp_every_steps is None else disp_every_steps
        eval_every_steps = len(train_loader) if eval_every_steps is None else eval_every_steps
        save_every_steps = np.inf if save_every_steps is None else save_every_steps
        if eval_every_steps % disp_every_steps != 0:
            raise ValueError(f"`eval_every_steps` {eval_every_steps} should be multiples of `disp_every_steps` {disp_every_steps}")
            
        self.model.train()
        
        best_eval_loss = np.inf
        # The `metric` must hold that it is better if higher, e.g., accuracy or F1. 
        best_eval_metric = 0.0
        
        train_losses = []
        train_y_gold, train_y_pred = [], []
        eidx, sidx = 0, 0
        done_training = False
        t0 = time.time()
        
        while eidx < n_epochs:
            for batch in train_loader:
                batch = batch.to(self.device)
                loss, *possible_batch_y = self.forward_batch(batch)
                self.backward_batch(loss)
                
                train_losses.append(loss.item())
                if possible_batch_y:
                    train_y_gold.extend(possible_batch_y[0])
                    train_y_pred.extend(possible_batch_y[1])
                
                if (sidx+1) % disp_every_steps == 0:
                    elapsed_secs = int(time.time() - t0)
                    if verbose:
                        max_lr = max(group['lr'] for group in self.optimizer.param_groups)
                        disp_running_info(eidx=eidx, sidx=sidx, lr=max_lr, 
                                          elapsed_secs=elapsed_secs, 
                                          loss=np.mean(train_losses),
                                          metric=self.evaluate(train_y_gold, train_y_pred) if train_y_gold else None,
                                          partition='train')
                    train_losses = []
                    train_y_gold, train_y_pred = [], []
                    t0 = time.time()
                
                if (sidx+1) % eval_every_steps == 0 and eval_loader is not None:
                    eval_loss, *possible_eval_metric = self.eval_epoch(eval_loader)
                    possible_eval_metric = possible_eval_metric[0] if possible_eval_metric else None
                    
                    elapsed_secs = int(time.time() - t0)
                    if verbose:
                        disp_running_info(elapsed_secs=elapsed_secs, 
                                          loss=eval_loss, 
                                          metric=possible_eval_metric, 
                                          partition='eval')
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if (save_callback is not None) and save_by_loss:
                            save_callback(self.model)
                        
                    if (possible_eval_metric is not None) and possible_eval_metric > best_eval_metric:
                        best_eval_metric = possible_eval_metric
                        if (save_callback is not None) and (not save_by_loss):
                            save_callback(self.model)
                    
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if save_by_loss:
                                assert self.scheduler.mode == 'min'
                                self.scheduler.step(eval_loss)
                            else:
                                assert self.scheduler.mode == 'max'
                                self.scheduler.step(possible_eval_metric)
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
            
            

def disp_running_info(eidx=None, sidx=None, lr=None, elapsed_secs=None, loss=None, metric=None, partition='train'):
    disp_text = []
    if eidx is not None:
        disp_text.append(f"Epoch: {eidx+1}")
    if sidx is not None:
        disp_text.append(f"Step: {sidx+1}")
    if lr is not None:
        disp_text.append(f"LR: {lr:.6f}")
    if len(disp_text) > 0:
        print(" | ".join(disp_text))
        
    if partition.lower().startswith('train'):
        partition = "Train"
    elif partition.lower().startswith('eval'):
        partition = "Eval."
    elif partition.lower().startswith('test'):
        partition = "Test "
    else:
        raise ValueError("Invalid partition {partition}")
        
    disp_text = []
    assert loss is not None
    disp_text.append(f"\t{partition} Loss: {loss:.3f}")
    if metric is not None:
        disp_text.append(f"{partition} Metric: {metric*100:.2f}%")
    else:
        disp_text.append(f"{partition} PPL: {np.exp(loss):.3f}")
    if elapsed_secs is not None:
        mins, secs = elapsed_secs // 60, elapsed_secs % 60
        disp_text.append(f"Elapsed Time: {mins}m {secs}s")
    print(" | ".join(disp_text))
    

