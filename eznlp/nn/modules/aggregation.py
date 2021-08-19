# -*- coding: utf-8 -*-
from typing import Union, List
import torch

from ..functional import sequence_pooling, sequence_group_aggregating
from ..init import reinit_layer_


class SequenceAttention(torch.nn.Module):
    """
    Attention over steps. 
    
    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, step)
    
    References
    ----------
    [1] D. Hu. 2018. An introductory survey on attention mechanisms in NLP problems. 
    [2] H. Chen, et al. 2016. Neural sentiment classification with user and product attention. 
    """
    def __init__(self, hid_dim: int, query_dim: int=None, scoring: str='Multiplicative'):
        super().__init__()
        if query_dim is None:
            query_dim = hid_dim
            
        self.query = torch.nn.Parameter(torch.empty(query_dim))
        uniform_range = (3 / query_dim) ** 0.5
        torch.nn.init.uniform_(self.query.data, -uniform_range, uniform_range)
        
        self.scoring = scoring
        if self.scoring.lower() == 'dot':
            if query_dim != hid_dim:
                raise ValueError(f"`query_dim` {query_dim} does not equals `hid_dim` {hid_dim}")
                
        elif self.scoring.lower() == 'multiplicative':
            self.proj_layer = torch.nn.Linear(hid_dim, query_dim)
            reinit_layer_(self.proj_layer, 'linear')
            
        elif self.scoring.lower() == 'additive':
            self.proj_layer = torch.nn.Linear(hid_dim+query_dim, query_dim)
            reinit_layer_(self.proj_layer, 'linear')
            
            self.w2 = torch.nn.Parameter(torch.empty(query_dim))
            uniform_range = (3 / query_dim) ** 0.5
            torch.nn.init.uniform_(self.w2.data, -uniform_range, uniform_range)
            
        else:
            raise ValueError(f"Invalid attention scoring mode {scoring}")
            
            
    def compute_scores(self, x: torch.FloatTensor):
        # x: (batch, step, hid_dim) -> scores: (batch, step)
        if self.scoring.lower() == 'dot':
            return x.matmul(self.query)
        elif self.scoring.lower() == 'multiplicative':
            # x: (batch, step, hid_dim) -> (batch, step, query_dim)
            x_projed = self.proj_layer(x)
            return x_projed.matmul(self.query)
        elif self.scoring.lower() == 'additive':
            # x: (batch, step, hid_dim) -> (batch, step, query_dim)
            x_query = torch.cat([x, self.query.expand(x.size(0), x.size(1), -1)], dim=-1)
            x_query_projed = self.proj_layer(x_query)
            return x_query_projed.matmul(self.w2)
            
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor=None, return_atten_weight: bool=False):
        # scores/atten_weight: (batch, step)
        scores = self.compute_scores(x)
        if mask is None:
            atten_weight = torch.nn.functional.softmax(scores, dim=-1)
        else:
            atten_weight = torch.nn.functional.softmax(scores.masked_fill(mask, float('-inf')), dim=-1)
        
        # atten_values: (batch, hid_dim)
        atten_values = atten_weight.unsqueeze(1).bmm(x).squeeze(1)
        
        if return_atten_weight:
            return atten_values, atten_weight
        else:
            return atten_values
        
    def __repr__(self):
        return f"{self.__class__.__name__}(scoring={self.scoring})"
    
    
class SequencePooling(torch.nn.Module):
    """
    Pooling values over steps. 
    
    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, step)
    mode: str
        'mean', 'max', 'min', 'wtd_mean'
    """
    def __init__(self, mode: str='mean'):
        super().__init__()
        if mode.lower() not in ('mean', 'max', 'min', 'wtd_mean'):
            raise ValueError(f"Invalid pooling mode {mode}")
        self.mode = mode
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor=None, weight: torch.FloatTensor=None):
        return sequence_pooling(x, mask, weight=weight, mode=self.mode)
    
    def extra_repr(self):
        return f"mode={self.mode}"
    
    
class SequenceGroupAggregating(torch.nn.Module):
    """
    Aggregating values over steps by groups. 
    
    Parameters
    ----------
    x : torch.FloatTensor (batch, ori_step, hidden)
        The tensor to be aggregate. 
    group_by : torch.LongTensor (batch, ori_step)
        The tensor indicating the positions after aggregation. 
        Positions being negative values are NOT used in aggregation. 
    agg_mode: str
        'mean', 'max', 'min', 'first', 'last'
    agg_step: int
    """
    def __init__(self, mode: str='mean'):
        super().__init__()
        if mode.lower() not in ('mean', 'max', 'min', 'first', 'last'):
            raise ValueError(f"Invalid aggregating mode {mode}")
        self.mode = mode
        
    def forward(self, x: torch.FloatTensor, group_by: torch.LongTensor, agg_step: int=None):
        return sequence_group_aggregating(x, group_by, agg_mode=self.mode, agg_step=agg_step)
    
    def extra_repr(self):
        return f"mode={self.mode}"




class ScalarMix(torch.nn.Module):
    """
    Mix multi-layer hidden states by corresponding scalar weights. 
    
    Computes a parameterised scalar mixture of N tensors, 
    ``mixture = gamma * \sum_k(s_k * tensor_k)``
    where ``s = softmax(w)``, with `w` and `gamma` scalar parameters.
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
    """
    def __init__(self, mix_dim: int):
        super().__init__()
        self.scalars = torch.nn.Parameter(torch.zeros(mix_dim))
        
    def __repr__(self):
        return f"{self.__class__.__name__}(mix_dim={self.scalars.size(0)})"
        
    def forward(self, tensors: Union[torch.FloatTensor, List[torch.FloatTensor]]):
        if isinstance(tensors, (list, tuple)):
            tensors = torch.stack(tensors)
        
        norm_weights_shape = tuple([-1] + [1] * (tensors.dim()-1))
        norm_weights = torch.nn.functional.softmax(self.scalars, dim=0).view(*norm_weights_shape)
        return (tensors * norm_weights).sum(dim=0)
