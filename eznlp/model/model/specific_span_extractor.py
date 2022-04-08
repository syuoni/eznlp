# -*- coding: utf-8 -*-
from typing import List, Union
from collections import OrderedDict
import torch

from ...config import ConfigList
from ...wrapper import Batch
from ..encoder import EncoderConfig
from ..decoder import (SingleDecoderConfigBase, 
                       SpecificSpanClsDecoderConfig, 
                       SpecificSpanRelClsDecoderConfig, 
                       SpecificSpanSparseRelClsDecoderConfig, 
                       JointExtractionDecoderConfig)
from .base import ModelConfigBase, ModelBase


class SpecificSpanExtractorConfig(ModelConfigBase):
    
    _pretrained_names = ['bert_like', 'span_bert_like']
    _all_names = _pretrained_names + ['intermediate2', 'span_intermediate2'] + ['decoder']
    
    def __init__(self, decoder: Union[SpecificSpanClsDecoderConfig, str]='specific_span_cls', **kwargs):
        self.bert_like = kwargs.pop('bert_like')
        self.span_bert_like = kwargs.pop('span_bert_like')
        self.intermediate2 = kwargs.pop('intermediate2', EncoderConfig(arch='LSTM', hid_dim=400))
        self.share_interm2 = kwargs.pop('share_interm2', True)
        
        if isinstance(decoder, (SingleDecoderConfigBase, JointExtractionDecoderConfig)):
            self.decoder = decoder
        elif isinstance(decoder, str):
            if decoder.lower().startswith('specific_span_cls'):
                self.decoder = SpecificSpanClsDecoderConfig()
            elif decoder.lower().startswith('specific_span_rel'):
                self.decoder = SpecificSpanRelClsDecoderConfig()
            elif decoder.lower().startswith('specific_span_sparse_rel'):
                self.decoder = SpecificSpanSparseRelClsDecoderConfig()
            elif decoder.lower().startswith('joint_extraction'):
                self.decoder = JointExtractionDecoderConfig(ck_decoder='specific_span_cls', rel_decoder='specific_span_rel_cls')
            else:
                raise ValueError(f"Invalid `decoder`: {decoder}")
        
        super().__init__(**kwargs)
        
        
    @property
    def valid(self):
        return super().valid and (self.bert_like is not None) and self.bert_like.output_hidden_states and (self.span_bert_like is not None)
        
    @property
    def span_intermediate2(self):
        if self.share_interm2:
            return None
        elif self.span_bert_like.share_weights_int:
            return self.intermediate2
        else:
            return ConfigList([self.intermediate2 for k in range(2, self.span_bert_like.max_span_size+1)])
        
        
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
        
        # `torch.nn.Module` use a set to keep up to one copy of each parameter/tensor, 
        # which is possibly shared and registered by different names
        # Hence, here we should manually avoid duplicate parameters/tensors
        if not self.span_bert_like.share_weights_ext:
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
                if not hasattr(self, 'span_intermediate2'):
                    new_all_last_query_states[k] = self.intermediate2(query_hidden, curr_mask)
                elif not isinstance(self.span_intermediate2, torch.nn.ModuleList):
                    new_all_last_query_states[k] = self.span_intermediate2(query_hidden, curr_mask)
                else:
                    new_all_last_query_states[k] = self.span_intermediate2[k-2](query_hidden, curr_mask)
            all_last_query_states = new_all_last_query_states
        
        return {'full_hidden': bert_hidden, 'all_query_hidden': all_last_query_states}
