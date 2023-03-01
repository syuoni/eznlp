# -*- coding: utf-8 -*-
from typing import List
from collections import OrderedDict
import torch
import transformers

from ..nn.modules import SequencePooling, SequenceAttention
from ..nn.modules import QueryBertLikeEncoder
from ..config import Config


class MaskedSpanBertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        self.hid_dim = self.bert_like.config.hidden_size
        
        self.arch = kwargs.pop('arch', 'BERT')
        self.freeze = kwargs.pop('freeze', True)
        
        self.num_layers = kwargs.pop('num_layers', None)
        if self.num_layers is None:
            self.num_layers = self.bert_like.config.num_hidden_layers
        assert 0 < self.num_layers <= self.bert_like.config.num_hidden_layers
        
        self.share_weights_ext = kwargs.pop('share_weights_ext', True)  # Share weights externally, i.e., with `bert_like`
        self.share_weights_int = kwargs.pop('share_weights_int', True)  # Share weights internally, i.e., between `query_bert_like` of different span sizes
        assert self.share_weights_int
        self.init_agg_mode = kwargs.pop('init_agg_mode', 'max_pooling')
        self.init_drop_rate = kwargs.pop('init_drop_rate', 0.2)
        super().__init__(**kwargs)
    
    @property
    def name(self):
        return f"{self.arch}({self.num_layers})"
        
    @property
    def out_dim(self):
        return self.hid_dim
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state
        
        
    def exemplify(self, cp_obj):
        # Attention mask for each example
        num_tokens, num_chunks = cp_obj.num_tokens, len(cp_obj.chunks)
        
        ck2tok_mask = torch.ones(num_chunks, num_tokens, dtype=torch.bool)
        for i, (label, start, end) in enumerate(cp_obj.chunks):
            for j in range(start, end):
                ck2tok_mask[i, j] = False
        return {'ck2tok_mask': ck2tok_mask}
        
        
    def batchify(self, batch_ex: List[dict], batch_sub_mask: torch.Tensor):
        # Batchify chunk-to-token attention mask
        # Remove `[CLS]` and `[SEP]` 
        batch_sub_mask = batch_sub_mask[:, 2:]
        max_num_chunks = max(ex['ck2tok_mask'].size(0) for ex in batch_ex)
        span_attention_mask = batch_sub_mask.unsqueeze(1).repeat(1, max_num_chunks, 1)
        ctx_attention_mask = batch_sub_mask.unsqueeze(1).repeat(1, max_num_chunks, 1)
        
        for i, ex in enumerate(batch_ex):
            ck2tok_mask = ex['ck2tok_mask']
            num_chunks, num_tokens = ck2tok_mask.size()
            
            # Assign to batched attention mask
            span_attention_mask[i, :num_chunks, :num_tokens].logical_or_(ck2tok_mask)
            ctx_attention_mask[i, :num_chunks, :num_tokens].logical_or_(~ck2tok_mask)
        
        return {'span_attention_mask': span_attention_mask, 
                'ctx_attention_mask': ctx_attention_mask}
        
        
    def instantiate(self):
        return MaskedSpanBertLikeEncoder(self)



class MaskedSpanBertLikeEncoder(torch.nn.Module):
    def __init__(self, config: MaskedSpanBertLikeConfig):
        super().__init__()
        if config.init_agg_mode.lower().endswith('_pooling'):
            self.init_aggregating = SequencePooling(mode=config.init_agg_mode.replace('_pooling', ''))
        elif config.init_agg_mode.lower().endswith('_attention'):
            self.init_aggregating = SequenceAttention(config.hid_dim, scoring=config.init_agg_mode.replace('_attention', ''))
        
        self.dropout = torch.nn.Dropout(config.init_drop_rate)
        
        assert config.share_weights_int
        # Share the module across all span sizes
        self.query_bert_like = QueryBertLikeEncoder(config.bert_like.encoder, num_layers=config.num_layers, share_weights=config.share_weights_ext)
        
        self.freeze = config.freeze
        self.num_layers = config.num_layers
        self.share_weights_ext = config.share_weights_ext
        self.share_weights_int = config.share_weights_int
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.query_bert_like.requires_grad_(not freeze)
        
        
    def get_extended_attention_mask(self, attention_mask: torch.Tensor):
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely. 
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.size()})")
        return attention_mask.float() * -10000
        
        
    def forward(self, all_hidden_states: List[torch.Tensor], span_attention_mask: torch.Tensor, ctx_attention_mask: torch.Tensor):
        # Remove the unused layers of hidden states
        all_hidden_states = all_hidden_states[-(self.num_layers+1):]
        batch_size, num_steps, hid_dim = all_hidden_states[0].size()
        
        all_last_query_states = tuple() 
        
        if isinstance(self.init_aggregating, (SequencePooling, SequenceAttention)): 
            for attention_mask in [span_attention_mask, ctx_attention_mask]: 
                # reshaped_states: (B, L, H) -> (B, NC, L, H) -> (B*NC, L, H)
                # attention_mask: (B, NC, L) -> (B*NC, L)
                reshaped_states0 = all_hidden_states[0].unsqueeze(1).expand(-1, attention_mask.size(1), -1, -1).contiguous().view(-1, num_steps, hid_dim)
                
                # query_states: (B*NC, H) -> (B, NC, H)
                query_states = self.init_aggregating(self.dropout(reshaped_states0), mask=attention_mask.view(-1, num_steps))
                query_states = query_states.view(batch_size, -1, hid_dim)
                
                query_outs = self.query_bert_like(query_states, all_hidden_states, self.get_extended_attention_mask(attention_mask))
                all_last_query_states += (query_outs['last_query_state'], )
        
        return all_last_query_states
