# -*- coding: utf-8 -*-
from typing import List
import torch

from ..data import Batch
from ..nn.init import reinit_layer_
from ..nn import CombinedDropout, SequencePooling
from ..config import Config


class DecoderConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.pooling = kwargs.pop('pooling', 'Max')
        if self.pooling.lower() not in ('max', 'mean'):
            raise ValueError(f"Invalid pooling method {self.pooling}")
            
        idx2label = kwargs.pop('idx2label', None)
        self.set_vocab(idx2label)
        super().__init__(**kwargs)
        
        
    def set_vocab(self, idx2label: List[str]):
        self.idx2label = idx2label
        self.label2idx = {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        
    def __repr__(self):
        repr_attr_dict = {key: self.__dict__[key] for key in ['in_dim', 'in_drop_rates']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def voc_dim(self):
        return len(self.label2idx)
        
    @property
    def pad_idx(self):
        return self.label2idx['<pad>']
    
    def instantiate(self):
        return Decoder(self)
        
    
        
class Decoder(torch.nn.Module):
    def __init__(self, config: DecoderConfig):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        self.pooling = SequencePooling(mode=config.pooling)
        
        self.hid2logit = torch.nn.Linear(config.in_dim, config.voc_dim)
        self.dropout = CombinedDropout(*config.in_drop_rates)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.idx2label = config.idx2label
        self.label2idx = config.label2idx
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.pooling(self.dropout(full_hidden), mask=batch.tok_mask)
        
        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        
        return self.criterion(logits, batch.label_id)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.pooling(full_hidden, mask=batch.tok_mask)
        
        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        
        return [self.idx2label[label_id] for label_id in logits.argmax(dim=-1).cpu().tolist()]
        
        
        