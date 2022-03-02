# -*- coding: utf-8 -*-
from typing import List
from collections import OrderedDict
import torch
import transformers

from ..nn.modules import SequencePooling, SequenceAttention
from ..nn.modules import QueryBertLikeEncoder
from ..config import Config


class SpanBertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        self.out_dim = self.bert_like.config.hidden_size
        self.num_layers = kwargs.pop('num_layers', self.bert_like.config.num_hidden_layers)
        assert 0 < self.num_layers <= self.bert_like.config.num_hidden_layers
        
        self.arch = kwargs.pop('arch', 'BERT')
        self.freeze = kwargs.pop('freeze', True)
        
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.share_weights = kwargs.pop('share_weights', False)
        self.init_agg_mode = kwargs.pop('init_agg_mode', 'max_pooling')
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self.arch
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state
        
    def instantiate(self):
        return SpanBertLikeEncoder(self)



class SpanBertLikeEncoder(torch.nn.Module):
    def __init__(self, config: SpanBertLikeConfig):
        super().__init__()
        if config.init_agg_mode.lower().endswith('_pooling'):
            self.init_aggregating = SequencePooling(mode=config.init_agg_mode.replace('_pooling', ''))
        elif config.init_agg_mode.lower().endswith('_attention'):
            self.init_aggregating = SequenceAttention(config.out_dim, scoring=config.init_agg_mode.replace('_attention', ''))
        
        if config.share_weights:
            # Share the module across all span sizes
            self.query_bert_like = QueryBertLikeEncoder(config.bert_like.encoder, num_layers=config.num_layers)
        else:
            # `ModuleDict` only accepts string keys. See https://github.com/pytorch/pytorch/issues/11714 
            # `self.query_bert_like[k-2]` for span size `k`
            self.query_bert_like = torch.nn.ModuleList([QueryBertLikeEncoder(config.bert_like.encoder, num_layers=config.num_layers) 
                                                            for _ in range(config.max_span_size-1)])
        
        self.freeze = config.freeze
        self.num_layers = config.num_layers
        self.max_span_size = config.max_span_size
        self.share_weights = config.share_weights
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.query_bert_like.requires_grad_(not freeze)
        
        
    def forward(self, all_hidden_states: List[torch.Tensor]):
        # Remove the unused layers of hidden states
        all_hidden_states = all_hidden_states[-(self.num_layers+1):]
        batch_size, num_steps, hid_dim = all_hidden_states[0].size()
        
        all_last_query_states = OrderedDict()
        for k in range(2, min(self.max_span_size, num_steps)+1):
            # reshaped_states: List of (B, L, H) -> (B, L-K+1, H, K) -> (B, L-K+1, K, H) -> (B*(L-K+1), K, H)
            reshaped_states = [hidden_states.unfold(dimension=1, size=k, step=1).permute(0, 1, 3, 2).flatten(end_dim=1) 
                                   for hidden_states in all_hidden_states]
            # query_states: (B*(L-K+1), 1, H)
            # query_states = reshaped_states[0].mean(dim=1, keepdim=True)
            query_states = self.init_aggregating(reshaped_states[0]).unsqueeze(1)
            
            if self.share_weights:
                query_outs = self.query_bert_like(query_states, reshaped_states)
            else:
                query_outs = self.query_bert_like[k-2](query_states, reshaped_states)  # `k` starts from 2
            
            # query_states: (B, L-K+1, H)
            all_last_query_states[k] = query_outs['last_query_state'].view(batch_size, -1, hid_dim)
        
        return all_last_query_states
