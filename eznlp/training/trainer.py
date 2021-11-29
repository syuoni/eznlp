# -*- coding: utf-8 -*-
import time
import numpy
import logging
import torch

from ..wrapper import Batch
from ..dataset import Dataset
from ..model.model import ModelBase

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Parameters
    ----------
    num_grad_acc_steps: int
        The "real" batch size is "nominal" `batch_size` * `num_grad_acc_steps`. 
    
    References
    ----------
    [1] https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    """
    def __init__(self, 
                 model: ModelBase, 
                 optimizer: torch.optim.Optimizer=None, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler=None, 
                 schedule_by_step: bool=False, 
                 num_grad_acc_steps: int=None, 
                 device: torch.device=None, 
                 non_blocking: bool=False,
                 grad_clip: float=None, 
                 use_amp: bool=False):
        self.model = model
        if hasattr(self.model, 'decoder'):
            self.num_metrics = self.model.decoder.num_metrics
        else:
            self.num_metrics = 0
        
        self.optimizer = optimizer
        if schedule_by_step:
            assert not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.scheduler = scheduler
        self.schedule_by_step = schedule_by_step
        self.num_grad_acc_steps = num_grad_acc_steps if isinstance(num_grad_acc_steps, int) and num_grad_acc_steps > 0 else 1
        self.num_steps = 0
        
        assert device is not None
        self.device = device
        self.non_blocking = non_blocking
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        
    def forward_batch(self, batch: Batch):
        """
        Forward to the loss (scalar). 
        Optionally return the predicted labels of the batch for evaluation. 
        
        Returns
        -------
        A scalar Tensor of loss, or
        A Tuple of (loss, y_pred_1, y_pred_2, ...)
        """
        losses, states = self.model(batch, return_states=True)
        loss = losses.mean()
        
        if self.num_metrics == 0:
            return loss
        else:
            return loss, *self.model.decoder._unsqueezed_decode(batch, **states)
        
        
    def backward_batch(self, loss: torch.Tensor):
        """
        Backward propagation and update the weights. 
        
        Parameters
        ----------
        loss: torch.Tensor
            A scalar tensor. 
        """
        # Average loss over the "real" batch. 
        # Note gradient accumulation is equivalent to summing the loss 
        loss = loss / self.num_grad_acc_steps

        # Notes: It is possible that the loss is calculated by tensors all with `requires_grad` being False;
        # e.g., in the span-based relation classification, all the examples in a batch have empty entity sets, 
        # then no negative pairs can be enumerated. 
        if loss.requires_grad:
            # Backward propagation
            self.scaler.scale(loss).backward()
        
        # `optimizer` follows the "real" steps
        self.num_steps += 1
        if self.num_steps % self.num_grad_acc_steps == 0:
            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # `scheduler` follows the "nominal" steps
        # `scheduler.step()` before `optimizer.step()` will raise warnings
        if self.scheduler is not None and self.schedule_by_step and self.num_steps >= self.num_grad_acc_steps:
            self.scheduler.step()
        
        
    def predict(self, dataset: Dataset, batch_size: int=32, beam_size: int=1):
        assert self.num_metrics == 1 or beam_size <= 1
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
        
        self.model.eval()
        set_y_pred = [[] for k in range(self.num_metrics)]
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                
                # `dataset` may not have ground-truths, so avoid computing loss here 
                if beam_size <= 1:
                    states = self.model.forward2states(batch)
                    batch_y_pred = self.model.decoder._unsqueezed_decode(batch, **states)
                    for k in range(self.num_metrics):
                        set_y_pred[k].extend(batch_y_pred[k])
                else:
                    # `num_metrics` must be 1
                    batch_y_pred = self.model.beam_search(beam_size, batch)
                    set_y_pred[0].extend(batch_y_pred)
        
        if self.num_metrics == 1:
            return set_y_pred[0]
        else:
            return set_y_pred
        
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
        Notes
        -----
        If `schedule_by_step` is False, `scheduler.step` should be properly called after this method. 
        """
        self.model.train()
        
        epoch_losses = []
        epoch_y_gold = [[] for k in range(self.num_metrics)]
        epoch_y_pred = [[] for k in range(self.num_metrics)]
        for batch in dataloader:
            batch = batch.to(self.device, non_blocking=self.non_blocking)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss_with_possible_y_pred = self.forward_batch(batch)
            
            if self.num_metrics == 0:
                loss = loss_with_possible_y_pred
            else:
                loss, *batch_y_pred = loss_with_possible_y_pred
                batch_y_gold = self.model.decoder._unsqueezed_retrieve(batch)
                
                for k in range(self.num_metrics):
                    epoch_y_gold[k].extend(batch_y_gold[k])
                    epoch_y_pred[k].extend(batch_y_pred[k])
            
            self.backward_batch(loss)
            epoch_losses.append(loss.item())
        
        if self.num_metrics == 0:
            return numpy.mean(epoch_losses)
        else:
            return numpy.mean(epoch_losses), *self.model.decoder._unsqueezed_evaluate(epoch_y_gold, epoch_y_pred)
        
        
        
    def eval_epoch(self, dataloader: torch.utils.data.DataLoader):
        self.model.eval()
        
        epoch_losses = []
        epoch_y_gold = [[] for k in range(self.num_metrics)]
        epoch_y_pred = [[] for k in range(self.num_metrics)]
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                loss_with_possible_y_pred = self.forward_batch(batch)
                
                if self.num_metrics == 0:
                    loss = loss_with_possible_y_pred
                else:
                    loss, *batch_y_pred = loss_with_possible_y_pred
                    batch_y_gold = self.model.decoder._unsqueezed_retrieve(batch)
                    
                    for k in range(self.num_metrics):
                        epoch_y_gold[k].extend(batch_y_gold[k])
                        epoch_y_pred[k].extend(batch_y_pred[k])
                
                epoch_losses.append(loss.item())
        
        if self.num_metrics == 0:
            return numpy.mean(epoch_losses)
        else:
            return numpy.mean(epoch_losses), *self.model.decoder._unsqueezed_evaluate(epoch_y_gold, epoch_y_pred)
    
    
    
    def train_steps(self, 
                    train_loader: torch.utils.data.DataLoader, 
                    dev_loader: torch.utils.data.DataLoader=None, 
                    num_epochs: int=10, 
                    max_steps: int=None, 
                    disp_every_steps: int=None, 
                    eval_every_steps: int=None, 
                    save_callback=None, 
                    save_by_loss: bool=True):
        """Train model by steps with optionally early-stop. 

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The loader of training data.
        dev_loader: torch.utils.data.DataLoader
            The loader of development data.
        num_epochs: int
            The number of epochs.
        max_steps: int
            The maximum number of steps.
        disp_every_steps: int
            Display running information by every `disp_every_steps` steps.
        eval_every_steps: int
            Evaluate and save the model by every `eval_every_steps` steps.
        save_callback
            The callback function to save model.
        save_by_loss: bool
            Whether to save by loss or other metrics. The metric must hold that it is better if higher, e.g., accuracy or F1. 
        """
        max_steps = numpy.inf if max_steps is None else max_steps
        disp_every_steps = len(train_loader) if disp_every_steps is None else disp_every_steps
        eval_every_steps = disp_every_steps  if eval_every_steps is None else eval_every_steps
        if eval_every_steps % disp_every_steps != 0:
            raise ValueError(f"`eval_every_steps` {eval_every_steps} should be multiples of `disp_every_steps` {disp_every_steps}")
        
        self.model.train()
        
        best_dev_loss = numpy.inf
        best_dev_metric = -numpy.inf
        
        train_losses = []
        train_y_gold = [[] for k in range(self.num_metrics)]
        train_y_pred = [[] for k in range(self.num_metrics)]
        eidx, sidx = 0, 0
        done_training = False
        t0 = time.time()
        
        while eidx < num_epochs:
            for batch in train_loader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss_with_possible_y_pred = self.forward_batch(batch)
                    
                if self.num_metrics == 0:
                    loss = loss_with_possible_y_pred
                else:
                    loss, *batch_y_pred = loss_with_possible_y_pred
                    batch_y_gold = self.model.decoder._unsqueezed_retrieve(batch)
                    
                    for k in range(self.num_metrics):
                        train_y_gold[k].extend(batch_y_gold[k])
                        train_y_pred[k].extend(batch_y_pred[k])
                    
                self.backward_batch(loss)
                train_losses.append(loss.item())
                
                if (sidx+1) % disp_every_steps == 0:
                    elapsed_secs = int(time.time() - t0)
                    lrs = [group['lr'] for group in self.optimizer.param_groups]
                    disp_running_info(eidx=eidx, sidx=sidx, lrs=lrs, 
                                      elapsed_secs=elapsed_secs, 
                                      loss=numpy.mean(train_losses),
                                      metric=self.model.decoder._unsqueezed_evaluate(train_y_gold, train_y_pred) if self.num_metrics>0 else None,
                                      partition='train')
                    train_losses = []
                    train_y_gold = [[] for k in range(self.num_metrics)]
                    train_y_pred = [[] for k in range(self.num_metrics)]
                    t0 = time.time()
                
                if (sidx+1) % eval_every_steps == 0 and dev_loader is not None:
                    loss_with_possible_metric = self.eval_epoch(dev_loader)
                    if self.num_metrics == 0:
                        dev_loss = loss_with_possible_metric
                    else:
                        dev_loss, *dev_metric = loss_with_possible_metric
                    
                    elapsed_secs = int(time.time() - t0)
                    disp_running_info(elapsed_secs=elapsed_secs, 
                                      loss=dev_loss, 
                                      metric=dev_metric if self.num_metrics>0 else None, 
                                      partition='dev')
                    
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        if (save_callback is not None) and save_by_loss:
                            save_callback(self.model)
                    
                    if self.num_metrics > 0 and numpy.mean(dev_metric) > best_dev_metric:
                        best_dev_metric = numpy.mean(dev_metric)
                        if (save_callback is not None) and (not save_by_loss):
                            save_callback(self.model)
                    
                    if self.scheduler is not None and not self.schedule_by_step:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if save_by_loss:
                                assert self.scheduler.mode == 'min'
                                self.scheduler.step(dev_loss)
                            else:
                                assert self.scheduler.mode == 'max'
                                self.scheduler.step(numpy.mean(dev_metric))
                        else:
                            self.scheduler.step()
                    
                    self.model.train()
                    t0 = time.time()
                
                if (sidx+1) % eval_every_steps == 0 and dev_loader is None:
                    # Always save the model if `dev_loader` is None
                    # Save multiple models by accordlingly defining `save_callback`
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
