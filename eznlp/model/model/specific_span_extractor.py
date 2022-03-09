# -*- coding: utf-8 -*-
from typing import List, Union
from collections import OrderedDict
import torch

from ...wrapper import Batch
from ..encoder import EncoderConfig
from ..decoder import SpecificSpanClsDecoderConfig
from .base import ModelConfigBase, ModelBase


class SpecificSpanExtractorConfig(ModelConfigBase):
    
    _pretrained_names = ['bert_like', 'span_bert_like']
    _all_names = _pretrained_names + ['intermediate2', 'intermediate3'] + ['decoder']
    
    def __init__(self, decoder: Union[SpecificSpanClsDecoderConfig, str]='specific_span', **kwargs):
        self.bert_like = kwargs.pop('bert_like', None)
        self.span_bert_like = kwargs.pop('span_bert_like', None)
        self.intermediate2 = kwargs.pop('intermediate2', EncoderConfig(arch='LSTM', hid_dim=400))
        self.intermediate3 = kwargs.pop('intermediate3', EncoderConfig(arch='FFN',  hid_dim=300, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        
        if isinstance(decoder, SpecificSpanClsDecoderConfig):
            self.decoder = decoder
        else:
            self.decoder = SpecificSpanClsDecoderConfig()
        
        super().__init__(**kwargs)
        
        
    @property
    def valid(self):
        return super().valid and (self.bert_like is not None) and self.bert_like.output_hidden_states and (self.span_bert_like is not None)
        
    def build_vocabs_and_dims(self, *partitions):
        curr_dim = self.bert_like.out_dim
        if self.intermediate2 is not None:
            self.intermediate2.in_dim, curr_dim = curr_dim, self.intermediate2.out_dim
        if self.intermediate3 is not None:
            self.intermediate3.in_dim, curr_dim = curr_dim, self.intermediate3.out_dim
        self.decoder.in_dim = curr_dim
        
        self.decoder.build_vocab(*partitions)
        self.span_bert_like.max_span_size = self.decoder.max_span_size
        
        
    def exemplify(self, entry: dict, training: bool=True):
        example = {}
        example['bert_like'] = self.bert_like.exemplify(entry['tokens'])
        example.update(self.decoder.exemplify(entry, training=training))
        return example
        
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch['bert_like'] = self.bert_like.batchify([ex['bert_like'] for ex in batch_examples])
        batch.update(self.decoder.batchify(batch_examples))
        return batch
        
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return SpecificSpanExtractor(self)



class SpecificSpanExtractor(ModelBase):
    def __init__(self, config: SpecificSpanExtractorConfig):
        super().__init__(config)
        if config.intermediate2 is not None:
            config.intermediate2.allow_empty = True
            if config.span_bert_like.share_weights:
                self.span_intermediate2 = config.intermediate2.instantiate()
            else:
                # `self.span_intermediate2[k-2]` for span size `k`
                self.span_intermediate2 = torch.nn.ModuleList([config.intermediate2.instantiate() 
                                                                   for _ in range(config.span_bert_like.max_span_size-1)])
        
        
    def pretrained_parameters(self):
        params = []
        params.extend(self.bert_like.bert_like.parameters())
        params.extend(self.span_bert_like.query_bert_like.parameters())
        return params
        
        
    def forward2states(self, batch: Batch):
        bert_hidden, all_bert_hidden = self.bert_like(**batch.bert_like)
        all_last_query_states = self.span_bert_like(all_bert_hidden)
        
        if hasattr(self, 'intermediate2'):
            bert_hidden = self.intermediate2(bert_hidden, batch.mask)
            
            new_all_last_query_states = OrderedDict()
            for k, query_hidden in all_last_query_states.items():
                # Allow empty sequences here; expect not to raise inconsistencies in final outputs
                curr_mask = batch.mask[:, k-1:].clone()
                curr_mask[:, 0] = False
                if self.span_bert_like.share_weights:
                    new_all_last_query_states[k] = self.span_intermediate2(query_hidden, curr_mask)
                else:
                    new_all_last_query_states[k] = self.span_intermediate2[k-2](query_hidden, curr_mask)
            all_last_query_states = new_all_last_query_states
        
        if hasattr(self, 'intermediate3'):
            # `intermediate3` is always shared across span sizes
            bert_hidden = self.intermediate3(bert_hidden, batch.mask)
            
            new_all_last_query_states = OrderedDict()
            for k, query_hidden in all_last_query_states.items():
                new_all_last_query_states[k] = self.intermediate3(query_hidden, batch.mask[:, k-1:])
            all_last_query_states = new_all_last_query_states
        
        return {'full_hidden': bert_hidden, 'all_query_hidden': all_last_query_states}
