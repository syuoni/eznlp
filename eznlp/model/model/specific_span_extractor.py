# -*- coding: utf-8 -*-
from typing import List, Union

from ...wrapper import Batch
from ..encoder import EncoderConfig
from ..decoder import SpecificSpanClsDecoderConfig
from .base import ModelConfigBase, ModelBase


class SpecificSpanExtractorConfig(ModelConfigBase):
    
    _pretrained_names = ['bert_like', 'span_bert_like']
    _all_names = _pretrained_names + ['intermediate2'] + ['decoder']
    
    def __init__(self, decoder: Union[SpecificSpanClsDecoderConfig, str]='specific_span', **kwargs):
        self.bert_like = kwargs.pop('bert_like', None)
        self.span_bert_like = kwargs.pop('span_bert_like', None)
        self.intermediate2 = kwargs.pop('intermediate2', EncoderConfig(arch='LSTM'))
        
        if isinstance(decoder, SpecificSpanClsDecoderConfig):
            self.decoder = decoder
        else:
            self.decoder = SpecificSpanClsDecoderConfig()
        
        super().__init__(**kwargs)
        
        
    @property
    def valid(self):
        return super().valid and (self.bert_like is not None) and self.bert_like.output_hidden_states and (self.span_bert_like is not None)
        
    def build_vocabs_and_dims(self, *partitions):
        if self.intermediate2 is not None:
            self.intermediate2.in_dim = self.bert_like.out_dim
            self.decoder.in_dim = self.intermediate2.out_dim
        else:
            self.decoder.in_dim = self.bert_like.out_dim
        
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
        
    def pretrained_parameters(self):
        params = []
        params.extend(self.bert_like.bert_like.parameters())
        params.extend(self.span_bert_like.query_bert_like.parameters())
        return params
        
    def forward2states(self, batch: Batch):
        bert_hidden, all_bert_hidden = self.bert_like(**batch.bert_like)
        all_last_query_states = self.span_bert_like(all_bert_hidden)
        return {'full_hidden': bert_hidden, 'all_query_hidden': all_last_query_states}
