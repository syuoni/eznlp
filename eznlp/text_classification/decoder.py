# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch

from ..data.wrapper import Batch
from ..nn.modules import SequencePooling, SequenceAttention
from ..model.decoder import DecoderConfig, Decoder



class TextClassificationDecoderConfig(DecoderConfig):
    def __init__(self, **kwargs):
        self.use_attention = kwargs.pop('use_attention', True)
        if self.use_attention:
            self.attention_scoring = kwargs.pop('attention_scoring', 'Multiplicative')
        else:
            self.pooling_mode = kwargs.pop('pooling_mode', 'Mean')
            
        self.idx2label = kwargs.pop('idx2label', None)
        super().__init__(**kwargs)
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def idx2label(self):
        return self._idx2label
    
    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        
    @property
    def voc_dim(self):
        return len(self.label2idx)
        
    @property
    def pad_idx(self):
        return self.label2idx['<pad>']
        
    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            counter.update([data_entry['label'] for data_entry in data])
        self.idx2label = list(counter.keys())
        
        
    def exemplify(self, data_entry: dict):
        return torch.tensor(self.label2idx[data_entry['label']])
        
    def batchify(self, batch_label_ids: list):
        return torch.stack(batch_label_ids)
        
    def instantiate(self):
        return TextClassificationDecoder(self)
        
    
    
class TextClassificationDecoder(Decoder):
    def __init__(self, config: TextClassificationDecoderConfig):
        super().__init__(config)
        if config.use_attention:
            self.pooling = SequenceAttention(config.in_dim, scoring=config.attention_scoring)
        else:
            self.pooling = SequencePooling(mode=config.pooling_mode)
            
        self.idx2label = config.idx2label
        self.label2idx = config.label2idx
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.pooling(self.dropout(full_hidden), mask=batch.tok_mask)
        
        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        
        return self.criterion(logits, batch.label_ids)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.pooling(full_hidden, mask=batch.tok_mask)
        
        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        
        return [self.idx2label[label_id] for label_id in logits.argmax(dim=-1).cpu().tolist()]
        
        