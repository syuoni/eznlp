# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def train_epoch(model, dataloader, device, optimizer, scheduler=None, clip=5):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        # Forward pass & Calculate loss
        batch = batch.to(device)
        loss, MLM_scores = model(input_ids=batch.MLM_wp_ids, 
                                 attention_mask=(~batch.attention_mask).type(torch.long), 
                                 labels=batch.MLM_wp_ids)
        
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
        
    return epoch_loss/len(dataloader)
    

def eval_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass & Calculate loss
            batch = batch.to(device)
            loss, MLM_scores = model(input_ids=batch.MLM_wp_ids, 
                                     attention_mask=(~batch.attention_mask).type(torch.long), 
                                     labels=batch.MLM_wp_ids)
            
            # Accumulate loss and acc
            epoch_loss += loss.item()
            
    return epoch_loss/len(dataloader)

