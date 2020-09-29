# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, model: nn.Module, optimizer=None, scheduler=None, 
                 device=None, grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        
        assert optimizer is not None
        assert device is not None
        
        
    def forward_batch(self, batch):
        """
        Forward to loss (scalar) and optionally compute accuracy.
        """
        raise NotImplementedError("Not Implemented `forward_batch`")
        
        
    def backward_batch(self, loss):
        """
        Backward propagation and update weights. 
        """
        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Update weights
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
    
    def train_epoch(self, dataloader):
        self.model.train()
        
        epoch_losses, epoch_accs = [], []
        for batch in dataloader:
            loss, possible_acc = self.forward_batch(batch)
            self.backward_batch(loss)
            
            epoch_losses.append(loss.item())
            epoch_accs.append(possible_acc)
            
        if any(epoch_accs):
            return np.mean(epoch_losses), np.mean(epoch_accs)
        else:
            return np.mean(epoch_losses), None
    
        
    def eval_epoch(self, dataloader):
        self.model.eval()
        
        epoch_losses, epoch_accs = [], []
        with torch.no_grad():
            for batch in dataloader:
                loss, possible_acc = self.forward_batch(batch)
                
                epoch_losses.append(loss.item())
                epoch_accs.append(possible_acc)
            
        if any(epoch_accs):
            return np.mean(epoch_losses), np.mean(epoch_accs)
        else:
            return np.mean(epoch_losses), None
    
    
    def train_steps(self, train_loader, eval_loader=None, n_epochs=10, max_steps=np.inf, 
                    disp_every_steps=500, eval_every_steps=1000, verbose=True, 
                    save_fn=None, save_by_loss=True):
        assert eval_every_steps % disp_every_steps == 0
        self.model.train()
        
        best_eval_loss = np.inf
        best_eval_acc = 0.0
        train_losses, train_accs = [], []
        eidx, sidx = 0, 0
        done_training = False
        t0 = time.time()
        
        while eidx < n_epochs:
            for batch in train_loader:
                loss, possible_acc = self.forward_batch(batch)
                self.backward_batch(loss)
                
                train_losses.append(loss.item())
                train_accs.append(possible_acc)
                
                if (sidx+1) % disp_every_steps == 0:
                    elapsed_secs = int(time.time() - t0)
                    if verbose:
                        disp_running_info(eidx=eidx, sidx=sidx, elapsed_secs=elapsed_secs, 
                                          loss=np.mean(train_losses),
                                          acc=np.mean(train_accs) if any(train_accs) else None,
                                          partition='train')
                    train_losses, train_accs = [], []
                    t0 = time.time()
                
                if (sidx+1) % eval_every_steps == 0:
                    if eval_loader is not None:
                        eval_loss, possible_eval_acc = self.eval_epoch(eval_loader)
                        elapsed_secs = int(time.time() - t0)
                        if verbose:
                            disp_running_info(elapsed_secs=elapsed_secs, 
                                              loss=eval_loss, 
                                              acc=possible_eval_acc, 
                                              partition='eval')
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            if (save_fn is not None) and save_by_loss:
                                torch.save(self.model, save_fn)
                            
                        if possible_eval_acc and possible_eval_acc > best_eval_acc:
                            best_eval_acc = possible_eval_acc
                            if (save_fn is not None) and (not save_by_loss):
                                torch.save(self.model, save_fn)
                        
                        self.model.train()
                        t0 = time.time()
                    
                if (sidx+1) >= max_steps:
                    done_training = True
                    break
                sidx += 1
            
            if done_training:
                break
            eidx += 1
            
            

def disp_running_info(eidx=None, sidx=None, elapsed_secs=None, loss=None, acc=None, partition='train'):
    disp_text = []
    if eidx is not None:
        disp_text.append(f"Epoch: {eidx+1}")
    if sidx is not None:
        disp_text.append(f"Step: {sidx+1}")
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
    if acc is not None:
        disp_text.append(f"{partition} Acc.: {acc*100:.2f}%")
    else:
        disp_text.append(f"{partition} PPL: {np.exp(loss):.3f}")
    if elapsed_secs is not None:
        mins, secs = elapsed_secs // 60, elapsed_secs % 60
        disp_text.append(f"Elapsed Time: {mins}m {secs}s")
    print(" | ".join(disp_text))
    
    

