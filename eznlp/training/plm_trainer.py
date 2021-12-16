# -*- coding: utf-8 -*-
import torch
import transformers
from .trainer import Trainer


class MaskedLMTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        
        
    def forward_batch(self, batch):
        batch_inputs = {'input_ids': batch.mlm_tok_ids, 
                        'attention_mask': (~batch.mlm_att_mask).long(), 
                        'labels': batch.mlm_lab_ids}
        
        if isinstance(self.model, transformers.BertForPreTraining):
            batch_inputs.update({'token_type_ids': batch.tok_type_ids, 
                                 'next_sentence_label': batch.paired_lab_ids})
        
        batch_outputs = self.model(**batch_inputs)
        loss = batch_outputs['loss']
        
        # In case of multi-GPU training
        if loss.dim() > 0:
            loss = loss.mean()
        
        return loss
