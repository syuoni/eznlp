# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from ..trainers import Trainer


class SequenceTaggingTrainer(Trainer):
    def __init__(self, model: nn.Module, optimizer=None, scheduler=None, 
                 device=None, grad_clip=1.0):
        super().__init__(model, optimizer=optimizer, scheduler=scheduler, 
                         device=device, grad_clip=grad_clip)
        
    def forward_batch(self, batch):
        batch = batch.to(self.device)
        losses, hidden = self.model(batch, return_hidden=True)
        loss = losses.mean()
        acc = calc_acc(self.model, batch, hidden)
        return loss, acc
        
        

def calc_acc(model, batch, hidden):
    best_paths = model.decode(batch, hidden)
    best_paths = [t for path in best_paths for t in path]
    # Here fetch the gold tags, instead of the modeling tags
    gold_paths = model.decoder.tag_helper.fetch_batch_tags(batch.tags_objs)
    gold_paths = [t for path in gold_paths for t in path]
    
    return sum(bt==gt for bt, gt in zip(best_paths, gold_paths)) / len(gold_paths)


def train_epoch(model, dataloader, device, optimizer, scheduler=None, clip=5):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in dataloader:
        # Forward pass & Calculate loss
        batch = batch.to(device)
        losses, hidden = model(batch, return_hidden=True)
        loss = losses.mean()
        
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), clip)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update weights
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Accumulate loss and acc
        epoch_loss += loss.item()
        epoch_acc  += calc_acc(model, batch, hidden)
        
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)
    

def eval_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass & Calculate loss
            batch = batch.to(device)
            losses, hidden = model(batch, return_hidden=True)
            loss = losses.mean()
            
            # Accumulate loss and acc
            epoch_loss += loss.item()
            epoch_acc  += calc_acc(model, batch, hidden)
            
    return epoch_loss/len(dataloader), epoch_acc/len(dataloader)

