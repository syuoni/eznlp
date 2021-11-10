# -*- coding: utf-8 -*-
import torch
from ..training import Trainer


class MaskedLMTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        mlm_outs = self.model(input_ids=batch.mlm_tok_ids, 
                              attention_mask=(~batch.attention_mask).long(), 
                              labels=batch.mlm_lab_ids)
        loss = mlm_outs['loss']
        
        # In case of Multi-GPU
        if loss.dim() > 0:
            loss = loss.mean()
            
        return loss
    
    