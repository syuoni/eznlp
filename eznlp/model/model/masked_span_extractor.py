# -*- coding: utf-8 -*-
from typing import List, Union
from collections import OrderedDict
import torch

from ...config import ConfigList
from ...wrapper import Batch
from ..encoder import EncoderConfig
from ..decoder import SingleDecoderConfigBase, MaskedSpanRelClsDecoderConfig
from .base import ModelConfigBase, ModelBase


class MaskedSpanExtractorConfig(ModelConfigBase):
    
    _pretrained_names = ['bert_like', 'masked_span_bert_like']
    _all_names = _pretrained_names + ['decoder']
    
    def __init__(self, decoder: Union[MaskedSpanRelClsDecoderConfig, str]='masked_span_rel', **kwargs):
        self.bert_like = kwargs.pop('bert_like')
        self.masked_span_bert_like = kwargs.pop('masked_span_bert_like')
        
        if isinstance(decoder, MaskedSpanRelClsDecoderConfig):
            self.decoder = decoder
        elif isinstance(decoder, str):
            if decoder.lower().startswith('masked_span_rel'):
                self.decoder = MaskedSpanRelClsDecoderConfig()
            else:
                raise ValueError(f"Invalid `decoder`: {decoder}")
        
        super().__init__(**kwargs)
        
        
    @property
    def valid(self):
        return super().valid and (self.bert_like is not None) and self.bert_like.output_hidden_states and (self.masked_span_bert_like is not None)
        
        
    def build_vocabs_and_dims(self, *partitions):
        self.decoder.in_dim = self.bert_like.out_dim
        self.decoder.build_vocab(*partitions)
        
        
    def exemplify(self, entry: dict, training: bool=True):
        example = {}
        example['bert_like'] = self.bert_like.exemplify(entry['tokens'])
        example.update(self.decoder.exemplify(entry, training=training))
        example['masked_span_bert_like'] = self.masked_span_bert_like.exemplify(example['cp_obj'])
        return example
        
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch['bert_like'] = self.bert_like.batchify([ex['bert_like'] for ex in batch_examples])
        batch.update(self.decoder.batchify(batch_examples))
        batch['masked_span_bert_like'] = self.masked_span_bert_like.batchify([ex['masked_span_bert_like'] for ex in batch_examples], batch['bert_like']['sub_mask'])
        return batch
        
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return MaskedSpanExtractor(self)

        
        
class MaskedSpanExtractor(ModelBase):
    def __init__(self, config: MaskedSpanExtractorConfig): 
        super().__init__(config)
        
        
    def pretrained_parameters(self):
        params = []
        params.extend(self.bert_like.bert_like.parameters())
        
        # `torch.nn.Module` use a set to keep up to one copy of each parameter/tensor, 
        # which is possibly shared and registered by different names
        # Hence, here we should manually avoid duplicate parameters/tensors
        if not self.masked_span_bert_like.share_weights_ext:
            params.extend(self.masked_span_bert_like.query_bert_like.parameters())
        
        return params
        
        
    def forward2states(self, batch: Batch):
        bert_hidden, all_bert_hidden = self.bert_like(**batch.bert_like)
        span_query_hidden, ctx_query_hidden = self.masked_span_bert_like(all_bert_hidden, **batch.masked_span_bert_like)
        
        return {'full_hidden': bert_hidden, 'span_query_hidden': span_query_hidden, 'ctx_query_hidden': ctx_query_hidden}
