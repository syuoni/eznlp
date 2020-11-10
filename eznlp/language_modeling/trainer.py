# -*- coding: utf-8 -*-
import torch

from ..trainer import Trainer


class MLMTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, optimizer=None, scheduler=None, 
                 device=None, grad_clip=1.0):
        super().__init__(model, optimizer=optimizer, scheduler=scheduler, 
                         device=device, grad_clip=grad_clip)
        
    def forward_batch(self, batch):
        loss, *_ = self.model(input_ids=batch.MLM_tok_ids, 
                              attention_mask=(~batch.attention_mask).type(torch.long), 
                              labels=batch.MLM_lab_ids)
        # In case of Multi-GPU
        if loss.dim() > 0:
            loss = loss.mean()
            
        # None as a placeholder for accuracy
        return loss, None
    