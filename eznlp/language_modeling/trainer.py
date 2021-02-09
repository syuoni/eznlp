# -*- coding: utf-8 -*-
import torch
from ..training.trainer import Trainer


class MLMTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        loss, *_ = self.model(input_ids=batch.MLM_tok_ids, 
                              attention_mask=(~batch.attention_mask).type(torch.long), 
                              labels=batch.MLM_lab_ids)
        # In case of Multi-GPU
        if loss.dim() > 0:
            loss = loss.mean()
            
        return (loss, )
    
    