# -*- coding: utf-8 -*-
import copy
import math
import torch
import transformers


class QueryBertLikeSelfAttention(torch.nn.Module):
    def __init__(self, origin: transformers.models.bert.modeling_bert.BertSelfAttention, share_weights: bool=False):
        super().__init__()
        self.num_attention_heads = origin.num_attention_heads
        self.attention_head_size = origin.attention_head_size
        self.all_head_size = origin.all_head_size
        
        if share_weights:
            self.query = origin.query
            self.key = origin.key
            self.value = origin.value
            self.dropout = origin.dropout
        else:
            # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483
            self.query = copy.deepcopy(origin.query)
            self.key = copy.deepcopy(origin.key)
            self.value = copy.deepcopy(origin.value)
            self.dropout = copy.deepcopy(origin.dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query_states, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(query_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs



class QueryBertLikeAttention(torch.nn.Module):
    def __init__(self, origin: transformers.models.bert.modeling_bert.BertAttention, share_weights: bool=False):
        super().__init__()
        self.self = QueryBertLikeSelfAttention(origin.self, share_weights=share_weights)
        if share_weights:
            self.output = origin.output
        else:
            self.output = copy.deepcopy(origin.output)
        
    def forward(self, query_states, hidden_states, **kwargs):
        self_outputs = self.self(query_states, hidden_states, **kwargs)
        attention_output = self.output(self_outputs[0], query_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



class QueryBertLikeLayer(torch.nn.Module):
    def __init__(self, origin: transformers.models.bert.modeling_bert.BertLayer, share_weights: bool=False):
        super().__init__()
        self.attention = QueryBertLikeAttention(origin.attention, share_weights=share_weights)
        if share_weights:
            self.intermediate = origin.intermediate
            self.output = origin.output
        else:
            self.intermediate = copy.deepcopy(origin.intermediate)
            self.output = copy.deepcopy(origin.output)
        
    def forward(self, query_states, hidden_states, **kwargs):
        self_attention_outputs = self.attention(query_states, hidden_states, **kwargs)
        attention_output = self_attention_outputs[0]
        
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        
        return outputs



class QueryBertLikeEncoder(torch.nn.Module):
    """A module of architecture corresponding to `BertEncoder`, where the `query_states` are different from `hidden_states`. 
    Typically, the `hidden_states` are pre-calculated by a `BertModel`. 
    To some extent, this module replaces the self-attention with cross-attention in `BertEncoder`. 
    
    References
    ----------
    transformers/models/bert/modeling_bert.py
    """
    def __init__(self, origin: transformers.models.bert.modeling_bert.BertEncoder, num_layers: int=None, share_weights: bool=False):
        super().__init__()
        if num_layers is None:
            num_layers = len(origin.layer)
        self.layer = torch.nn.ModuleList([QueryBertLikeLayer(layer, share_weights=share_weights) for layer in origin.layer[-num_layers:]])
        
        
    def forward(self, 
                query_states, 
                all_hidden_states, 
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                output_query_states=False, 
                return_dict=True):
        assert len(self.layer) + 1 == len(all_hidden_states)
        
        all_query_states = () if output_query_states else None
        all_self_attentions = () if output_attentions else None
        for i, (layer_module, hidden_states) in enumerate(zip(self.layer, all_hidden_states[:-1])):
            if output_query_states:
                all_query_states = all_query_states + (query_states,)
            
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(query_states, 
                                         hidden_states, 
                                         attention_mask=attention_mask, 
                                         head_mask=layer_head_mask, 
                                         output_attentions=output_attentions)
            # Update query_states to layer+1
            query_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        
        if output_query_states:
            all_query_states = all_query_states + (query_states,)
        
        if not return_dict:
            return tuple(v for v in [query_states, all_query_states, all_self_attentions])
        else:
            return dict(last_query_state=query_states, query_states=all_query_states, attentions=all_self_attentions)
