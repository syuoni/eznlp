# -*- coding: utf-8 -*-
import torch

from ..init import reinit_layer_, reinit_vector_parameter_


class SequenceAttention(torch.nn.Module):
    """Attention over steps. 
    
    Notes
    -----
    * If `external_query` is True, the query vector is an inside parameter of this module, 
    and this module works as an aggregating layer. 
    * If `external_query` is False, the query vector should be passed from outside. 

    Parameters
    ----------
    x: torch.FloatTensor (batch, key_step, key_dim)
        The key/values sequence.
    mask: torch.BoolTensor (batch, key_step)
        The padding mask of key/values sequence.
    query: torch.FloatTensor (batch, query_step, query_dim)
        The query sequence.

    References
    ----------
    [1] D. Hu. 2018. An introductory survey on attention mechanisms in NLP problems. 
    [2] H. Chen, et al. 2016. Neural sentiment classification with user and product attention. 
    [3] A. Vaswani, et al. 2018. Attention is all you need. 
    """
    def __init__(self, key_dim: int, query_dim: int=None, atten_dim: int=None, 
                 scoring: str='additive', nonlinearity: str='tanh', 
                 external_query: bool=False):
        super().__init__()
        if query_dim is None:
            query_dim = key_dim
        if atten_dim is None:
            atten_dim = key_dim
        
        if not external_query:
            self.query = torch.nn.Parameter(torch.empty(query_dim))
            reinit_vector_parameter_(self.query)
        
        if scoring.lower() in ('dot', 'scaled_dot'):
            assert query_dim == key_dim, f"`query_dim` {query_dim} does not equals `key_dim` {key_dim}"
            
        elif scoring.lower() == 'multiplicative':
            self.proj_layer = torch.nn.Linear(key_dim, query_dim)
            reinit_layer_(self.proj_layer, nonlinearity)
            
        elif scoring.lower() == 'additive':
            self.proj_layer = torch.nn.Linear(key_dim+query_dim, atten_dim)
            reinit_layer_(self.proj_layer, nonlinearity)
            
            self.w2 = torch.nn.Parameter(torch.empty(atten_dim))
            reinit_vector_parameter_(self.w2)
            
        elif scoring.lower() == 'biaffine':
            self.query_proj_layer = torch.nn.Linear(query_dim, atten_dim)
            self.key_proj_layer = torch.nn.Linear(key_dim, atten_dim)
            reinit_layer_(self.query_proj_layer, nonlinearity)
            reinit_layer_(self.key_proj_layer, nonlinearity)
            
            self.w2 = torch.nn.Parameter(torch.empty(atten_dim))
            reinit_vector_parameter_(self.w2)
            
        else:
            raise ValueError(f"Invalid attention scoring mode {scoring}")
        
        if nonlinearity.lower() == 'linear':
            self.activation = torch.nn.Identidy()
        elif nonlinearity.lower() == 'tanh':
            self.activation = torch.nn.Tanh()
        elif nonlinearity.lower() == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            raise ValueError(f"Invalid nonlinearity {nonlinearity}")
        
        self.scoring = scoring
        self.nonlinearity = nonlinearity
        
        
    def compute_scores(self, query: torch.FloatTensor, x: torch.FloatTensor):
        if query.dim() == 1:
            # query: (query_dim, ) -> (batch, query_step=1, query_dim)
            query = query.expand(x.size(0), 1, -1)
        elif query.dim() == 2:
            # query: (batch, query_dim) -> (batch, query_step=1, query_dim)
            query = query.unsqueeze(1)
        
        # x: (batch, key_step, key_dim) -> scores: (batch, query_step, key_step)
        if self.scoring.lower() == 'dot':
            return query.bmm(x.permute(0, 2, 1))
            
        elif self.scoring.lower() == 'scaled_dot':
            return query.bmm(x.permute(0, 2, 1)) / (query.size(-1) ** 0.5)
            
        elif self.scoring.lower() == 'multiplicative':
            # x_projed: (batch, key_step, query_dim)
            x_projed = self.activation(self.proj_layer(x))
            return query.bmm(x_projed.permute(0, 2, 1))
            
        elif self.scoring.lower() == 'additive':
            # x_query: (batch, query_step, key_step, key_dim+query_dim)
            x_query = torch.cat([x.unsqueeze(1).expand(-1, query.size(1), -1, -1), 
                                 query.unsqueeze(2).expand(-1, -1, x.size(1), -1)], dim=-1)
            x_query_projed = self.activation(self.proj_layer(x_query))
            return x_query_projed.matmul(self.w2)
            
        elif self.scoring.lower() == 'biaffine':
            # x_query: (batch, query_step, key_step, atten_dim)
            x_query = self.key_proj_layer(x).unsqueeze(1) + self.query_proj_layer(query).unsqueeze(2)
            return self.activation(x_query).matmul(self.w2)
        
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor=None, query: torch.FloatTensor=None, return_atten_weight: bool=False):
        if hasattr(self, 'query'):
            assert query is None
            query = self.query
        else:
            assert query is not None
        scores = self.compute_scores(query, x)
        
        # x: (batch, key_step, key_dim)
        # scores/atten_weight: (batch, query_step, key_step)
        # mask: (batch, key_step)
        if mask is None:
            atten_weight = torch.nn.functional.softmax(scores, dim=-1)
        else:
            atten_weight = torch.nn.functional.softmax(scores.masked_fill(mask.unsqueeze(1), float('-inf')), dim=-1)
        
        # atten_values: (batch, query_step, key_dim)
        atten_values = atten_weight.bmm(x)
        
        if query.dim() <= 2:
            atten_values = atten_values.squeeze(1)
            atten_weight = atten_weight.squeeze(1)
        
        if return_atten_weight:
            return atten_values, atten_weight
        else:
            return atten_values
        
        
    def __repr__(self):
        return f"{self.__class__.__name__}(scoring={self.scoring}, nonlinearity={self.nonlinearity})"
