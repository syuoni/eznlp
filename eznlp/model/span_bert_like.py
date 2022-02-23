# -*- coding: utf-8 -*-
import torch
import transformers

from ..nn.modules import QueryBertLikeEncoder
from ..config import Config


class SpanBertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        self.out_dim = self.bert_like.config.hidden_size
        self.num_layers = self.bert_like.config.num_hidden_layers
        
        # num_layers
        self.max_span_size = kwargs.pop('max_span_size', 10)
        self.share_weights = kwargs.pop('share_weights', False)
        self.init_agg_mode = kwargs.pop('init_agg_mode', 'max')
        super().__init__(**kwargs)
        
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state
        
    def instantiate(self):
        return SpanBertLikeEncoder(self)



class SpanBertLikeEncoder(torch.nn.Module):
    def __init__(self, config: SpanBertLikeConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
        # `ModuleDict` only accepts string keys. 
        # See https://github.com/pytorch/pytorch/issues/11714 
        self.query_bert_like = torch.nn.ModuleList([QueryBertLikeEncoder(config.bert_like.encoder) for _ in range(config.max_span_size-1)])
        
        
    def forward(self, all_hidden_states):
        batch_size, num_steps, hid_dim = all_hidden_states[0].size()
        
        all_last_query_states = {}
        for k in range(2, self.max_span_size+1):
            if k <= num_steps:
                # reshaped_states: List of (B*(L-K+1), K, H)
                reshaped_states = [hidden_states.unfold(dimension=1, size=k, step=1).permute(0, 1, 3, 2).flatten(end_dim=1) 
                                    for hidden_states in all_hidden_states]
                # query_states: (B*(L-K+1), 1, H)
                query_states = reshaped_states[0].mean(dim=1, keepdim=True)
                query_outs = self.query_bert_like[k-2](query_states, reshaped_states)  # `k` starts from 2
                all_last_query_states[k] = query_outs['last_query_state'].view(batch_size, -1, hid_dim)
            else:
                break
        
        return all_last_query_states
