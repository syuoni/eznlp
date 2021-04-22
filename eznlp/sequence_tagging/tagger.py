# -*- coding: utf-8 -*-
from typing import List
import torch

from ..data.wrapper import Batch
from ..model.model import ModelConfig, Model
from .decoder import SequenceTaggingDecoderConfig


class SequenceTaggerConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.decoder: SequenceTaggingDecoderConfig = kwargs.pop('decoder', SequenceTaggingDecoderConfig())
        super().__init__(**kwargs)
    
    @property
    def valid(self):
        if self.bert_like is not None and not self.bert_like.from_tokenized:
            return False
        return super().valid and (self.decoder is not None) and self.decoder.valid
        
    @property
    def name(self):
        return self._name_sep.join([super().name, self.decoder.arch])
        
    def build_vocabs_and_dims(self, *partitions):
        super().build_vocabs_and_dims(*partitions)
        
        if self.intermediate2 is not None:
            self.decoder.in_dim = self.intermediate2.out_dim
        else:
            self.decoder.in_dim = self.full_hid_dim
            
        self.decoder.build_vocab(*partitions)
        
        
    def exemplify(self, data_entry: dict, training: bool=True):
        example = super().exemplify(data_entry['tokens'])
        example['tags_obj'] = self.decoder.exemplify(data_entry, training=training)
        return example
        
    
    def batchify(self, batch_examples: List[dict]):
        batch = super().batchify(batch_examples)
        batch['tags_objs'] = self.decoder.batchify([ex['tags_obj'] for ex in batch_examples])
        return batch
        
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return SequenceTagger(self)
    
    
class SequenceTagger(Model):
    def __init__(self, config: SequenceTaggerConfig):
        super().__init__(config)
        
    def forward(self, batch: Batch, return_hidden: bool=False):
        full_hidden = self.get_full_hidden(batch)
        losses = self.decoder(batch, full_hidden)
        
        # Return `hidden` for the `decode` method, to avoid duplicated computation. 
        if return_hidden:
            return losses, full_hidden
        else:
            return losses
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor=None):
        if full_hidden is None:
            full_hidden = self.get_full_hidden(batch)
            
        return self.decoder.decode(batch, full_hidden)
        