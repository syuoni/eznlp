# -*- coding: utf-8 -*-
import torch

from ..init import reinit_layer_, reinit_vector_parameter_
from ..utils import _nonlinearity2activation


# TODO: Scaling factor for other scoring?

class SequenceAttention(torch.nn.Module):
    """Attention over steps. 
    
    Notes
    -----
    * If `external_query` is True, the query vector is an inside parameter of this module, 
    and this module works as an aggregating layer. 
    * If `external_query` is False, the query vector should be passed from outside. 
    
    Parameters
    ----------
    x: torch.Tensor (batch, key_step, value_dim)
        The value sequence.
    mask: torch.Tensor (batch, key_step)
        The padding mask of key/value sequence.
    query: torch.Tensor (batch, query_step, query_dim)
        The query sequence.
    key: torch.Tensor (batch, key_step, key_dim)
        The key sequence. 
    
    References
    ----------
    [1] D. Hu. 2018. An introductory survey on attention mechanisms in NLP problems. 
    [2] H. Chen, et al. 2016. Neural sentiment classification with user and product attention. 
    [3] A. Vaswani, et al. 2018. Attention is all you need. 
    """
    def __init__(self, key_dim: int, query_dim: int=None, atten_dim: int=None, num_heads: int=1, 
                 scoring: str='additive', nonlinearity: str='tanh', drop_rate: float=0.0, external_query: bool=False):
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
            self.proj_layer = torch.nn.Linear(key_dim//num_heads, query_dim//num_heads)
            reinit_layer_(self.proj_layer, nonlinearity)
            
        elif scoring.lower() == 'additive':
            self.proj_layer = torch.nn.Linear((key_dim+query_dim)//num_heads, atten_dim)
            reinit_layer_(self.proj_layer, nonlinearity)
            
            self.w2 = torch.nn.Parameter(torch.empty(atten_dim))
            reinit_vector_parameter_(self.w2)
            
        elif scoring.lower() == 'biaffine':
            self.query_proj_layer = torch.nn.Linear(query_dim//num_heads, atten_dim)
            self.key_proj_layer = torch.nn.Linear(key_dim//num_heads, atten_dim)
            reinit_layer_(self.query_proj_layer, nonlinearity)
            reinit_layer_(self.key_proj_layer, nonlinearity)
            
            self.w2 = torch.nn.Parameter(torch.empty(atten_dim))
            reinit_vector_parameter_(self.w2)
            
        else:
            raise ValueError(f"Invalid attention scoring mode {scoring}")
        
        self.activation = _nonlinearity2activation(nonlinearity)
        self.dropout = torch.nn.Dropout(drop_rate)
        
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.scoring = scoring
        self.nonlinearity = nonlinearity
        
        
    def compute_scores(self, query: torch.Tensor, key: torch.Tensor):
        # query: (batch, query_step, query_dim)
        # key: (batch, key_step, key_dim) 
        # scores: (batch, query_step, key_step)
        if self.scoring.lower() == 'dot':
            return query.bmm(key.permute(0, 2, 1))
            
        elif self.scoring.lower() == 'scaled_dot':
            return query.bmm(key.permute(0, 2, 1)) / (query.size(-1) ** 0.5)
            
        elif self.scoring.lower() == 'multiplicative':
            # key_projed: (batch, key_step, query_dim)
            key_projed = self.activation(self.proj_layer(key))
            return query.bmm(key_projed.permute(0, 2, 1))
            
        elif self.scoring.lower() == 'additive':
            # key_query: (batch, query_step, key_step, key_dim+query_dim)
            key_query = torch.cat([key.unsqueeze(1).expand(-1, query.size(1), -1, -1), 
                                   query.unsqueeze(2).expand(-1, -1, key.size(1), -1)], dim=-1)
            key_query_projed = self.activation(self.proj_layer(key_query))
            return key_query_projed.matmul(self.w2)
            
        elif self.scoring.lower() == 'biaffine':
            # key_query: (batch, query_step, key_step, atten_dim)
            key_query = self.key_proj_layer(key).unsqueeze(1) + self.query_proj_layer(query).unsqueeze(2)
            return self.activation(key_query).matmul(self.w2)
        
        
    def _prepare_multiheads(self, x: torch.Tensor):
        assert x.size(-1) % self.num_heads == 0
        batch_size = x.size(0)
        dim_per_head = x.size(-1) // self.num_heads
        # x: (batch, step, dim) -> (batch*num_heads, step, dim/num_heads)
        return x.view(batch_size, -1, self.num_heads, dim_per_head).permute(0, 2, 1, 3).contiguous().view(batch_size*self.num_heads, -1, dim_per_head)
        
    def _restore_multiheads(self, x: torch.Tensor):
        batch_size = x.size(0) // self.num_heads
        dim_per_head = x.size(-1)
        # x: (batch*num_heads, step, dim/num_heads) -> (batch, step, dim)
        return x.view(batch_size, self.num_heads, -1, dim_per_head).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, dim_per_head*self.num_heads)
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, query: torch.Tensor=None, key: torch.Tensor=None, return_atten_weight: bool=False):
        if hasattr(self, 'query'):
            assert query is None
            query = self.query
        else:
            assert query is not None
        
        original_query_num_dims = query.dim()
        if original_query_num_dims == 1:
            # query: (query_dim, ) -> (batch, query_step=1, query_dim)
            query = query.expand(x.size(0), 1, -1)
        elif original_query_num_dims == 2:
            # query: (batch, query_dim) -> (batch, query_step=1, query_dim)
            query = query.unsqueeze(1)
        
        if key is None:
            key = x
        
        # x/key: (batch, key_step, value_dim/key_dim)
        # query: (batch, query_step, query_dim)
        # mask: (batch, key_step) or (batch, query_step, key_step)
        
        if self.num_heads > 1:
            query = self._prepare_multiheads(query)
            key   = self._prepare_multiheads(key)
            x     = self._prepare_multiheads(x)
        
        # scores/atten_weight: (batch, query_step, key_step)
        scores = self.compute_scores(query, key)
        
        if mask is None:
            atten_weight = torch.nn.functional.softmax(scores, dim=-1)
        else:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            assert mask.dim() == 3
            if self.num_heads > 1:
                mask  = mask.repeat_interleave(self.num_heads, dim=0)
            atten_weight = torch.nn.functional.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
            atten_weight = self.dropout(atten_weight) # Apply dropout on attention weight?
        
        # atten_values: (batch, query_step, value_dim)
        atten_values = atten_weight.bmm(x)
        
        if self.num_heads > 1:
            atten_values = self._restore_multiheads(atten_values)
            atten_weight = atten_weight.view(atten_values.size(0), self.num_heads, atten_weight.size(-2), atten_weight.size(-1))
        
        if original_query_num_dims <= 2:
            atten_values = atten_values.squeeze(-2)
            atten_weight = atten_weight.squeeze(-2)
        
        # atten_values: (batch, query_step, value_dim)
        # atten_weight: (batch, query_step, key_step) or (batch, num_heads, query_step, key_step)
        if return_atten_weight:
            return atten_values, atten_weight
        else:
            return atten_values
        
        
    def __repr__(self):
        return f"{self.__class__.__name__}(key_dim={self.key_dim}, query_dim={self.query_dim}, num_heads={self.num_heads}, scoring={self.scoring}, nonlinearity={self.nonlinearity})"



class MultiheadAttention(torch.nn.Module):
    def __init__(self, query_dim: int, key_dim: int=None, value_dim: int=None, affine_dim: int=None, out_dim: int=None, num_heads: int=8, 
                 scoring: str='scaled_dot', drop_rate: float=0.1, **kwargs):
        super().__init__()
        if key_dim is None:
            key_dim = query_dim
        if value_dim is None:
            value_dim = query_dim
        if affine_dim is None:
            affine_dim = query_dim
        if out_dim is None:
            out_dim = query_dim
        
        self.query_affine = torch.nn.Linear(query_dim, affine_dim)
        self.key_affine   = torch.nn.Linear(key_dim, affine_dim)
        self.value_affine = torch.nn.Linear(value_dim, affine_dim)
        reinit_layer_(self.query_affine, 'linear')
        reinit_layer_(self.key_affine, 'linear')
        reinit_layer_(self.value_affine, 'linear')
        
        self.attention = SequenceAttention(key_dim=affine_dim, query_dim=affine_dim, num_heads=num_heads, 
                                           scoring=scoring, drop_rate=drop_rate, external_query=True, **kwargs)
        self.out_affine = torch.nn.Linear(affine_dim, out_dim)
        reinit_layer_(self.out_affine, 'linear')
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor=None, return_atten_weight: bool=False):
        QW = self.query_affine(query)
        KW = self.key_affine(key)
        VW = self.value_affine(value)
        
        atten_values, atten_weight = self.attention(VW, mask=mask, query=QW, key=KW, return_atten_weight=True)
        OW = self.out_affine(atten_values)
        if return_atten_weight:
            return OW, atten_weight
        else:
            return OW
