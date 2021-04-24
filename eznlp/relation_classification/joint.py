# -*- coding: utf-8 -*-
from typing import List
import torch

from ..data.wrapper import Batch
from ..model.model import ModelConfig, Model
from ..model.decoder import DecoderConfig
from ..span_classification.decoder import SpanClassificationDecoderConfig
from .decoder import RelationClassificationDecoderConfig


class JointClassifierConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.ck_decoder: DecoderConfig = kwargs.pop('ck_decoder', SpanClassificationDecoderConfig())
        self.rel_decoder: DecoderConfig = kwargs.pop('rel_decoder', RelationClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def valid(self):
        if self.bert_like is not None and not self.bert_like.from_tokenized:
            return False
        return super().valid and (self.ck_decoder is not None) and self.ck_decoder.valid and (self.rel_decoder is not None) and self.rel_decoder.valid
    
    @property
    def name(self):
        return self._name_sep.join([super().name, self.ck_decoder.name, self.rel_decoder.name])
    
    def build_vocabs_and_dims(self, *partitions):
        super().build_vocabs_and_dims(*partitions)
        
        if self.intermediate2 is not None:
            self.ck_decoder.in_dim = self.intermediate2.out_dim
            self.rel_decoder.in_dim = self.intermediate2.out_dim
        else:
            self.ck_decoder.in_dim = self.full_hid_dim
            self.rel_decoder.in_dim = self.full_hid_dim
            
        self.ck_decoder.build_vocab(*partitions)
        self.rel_decoder.build_vocab(*partitions)
    
    
    def exemplify(self, data_entry: dict, training: bool=True):
        example = super().exemplify(data_entry['tokens'])
        example['spans_obj'] = self.ck_decoder.exemplify(data_entry, training=training)
        example['span_pairs_obj'] = self.rel_decoder.exemplify(data_entry, training=training)
        return example
    
    def batchify(self, batch_examples: List[dict]):
        batch = super().batchify(batch_examples)
        batch['spans_objs'] = self.ck_decoder.batchify([ex['spans_obj'] for ex in batch_examples])
        batch['span_pairs_objs'] = self.rel_decoder.batchify([ex['span_pairs_obj'] for ex in batch_examples])
        return batch
    
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return JointClassifier(self)
    
    
    
class JointClassifier(Model):
    def __init__(self, config: JointClassifierConfig):
        super().__init__(config)
        
    def forward(self, batch: Batch, return_hidden: bool=False):
        full_hidden = self.get_full_hidden(batch)
        ck_losses = self.ck_decoder(batch, full_hidden)
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        
        for k in range(batch.size):
            batch.span_pairs_objs[k].build(batch_chunks_pred[k]).to(full_hidden.device)
        rel_losses = self.rel_decoder(batch, full_hidden)
        
        # Return `hidden` for the `decode` method, to avoid duplicated computation. 
        if return_hidden:
            return (ck_losses, rel_losses), full_hidden
        else:
            return (ck_losses, rel_losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor=None):
        if full_hidden is None:
            full_hidden = self.get_full_hidden(batch)
            
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        
        if not batch.span_pairs_objs[0].is_built:
            for k in range(batch.size):
                batch.span_pairs_objs[k].build(batch_chunks_pred[k]).to(full_hidden.device)
        batch_relations_pred = self.rel_decoder.decode(batch, full_hidden)
        
        return (batch_chunks_pred, batch_relations_pred)
    
    