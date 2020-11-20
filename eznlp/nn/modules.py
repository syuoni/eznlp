# -*- coding: utf-8 -*-
import torch
from .functional import sequence_pooling, sequence_group_aggregating


class SequencePooling(torch.nn.Module):
    """
    Pooling values over steps. 
    
    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, step)
    mode: str
        'mean', 'max', 'min'
    """
    def __init__(self, mode: str='mean'):
        super().__init__()
        if mode.lower() not in ('mean', 'max', 'min'):
            raise ValueError(f"Invalid pooling mode {mode}")
        self.mode = mode
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        return sequence_pooling(x, mask, mode=self.mode)
    
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
    
    
    
class CombinedDropout(torch.nn.Module):
    def __init__(self, p: float=0.0, word_p: float=0.05, locked_p: float=0.5):
        super().__init__()
        if p > 0:
            self.dropout = torch.nn.Dropout(p)
        if word_p > 0:
            self.word_dropout = WordDropout(word_p)
        if locked_p > 0:
            self.locked_dropout = LockedDropout(locked_p)
        
    def forward(self, x: torch.Tensor):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        if hasattr(self, 'word_dropout'):
            x = self.word_dropout(x)
        if hasattr(self, 'locked_dropout'):
            x = self.locked_dropout(x)
        return x


class LockedDropout(torch.nn.Module):
    """
    Locked (or variational) dropout, which drops out all elements (over steps) 
    in randomly chosen dimension of embeddings or hidden states. 
    
    References
    ----------
    https://github.com/flairNLP/flair/blob/master/flair/nn.py
    """
    def __init__(self, p: float=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        
    def forward(self, x: torch.Tensor):
        if (not self.training) or self.p == 0:
            return x
        
        # x: (batch, step, hidden)
        m = torch.empty(x.size(0), 1, x.size(2), device=x.device).bernoulli(p=1-self.p)
        return x * m / (1 - self.p)
    
    def extra_repr(self):
        return f"p={self.p}"


class WordDropout(torch.nn.Module):
    """
    Word dropout, which drops out all elements (over embeddings or hidden states)
    in randomly chosen word. 
    
    References
    ----------
    https://github.com/flairNLP/flair/blob/master/flair/nn.py
    """
    def __init__(self, p: float=0.05):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        
    def forward(self, x: torch.Tensor):
        if (not self.training) or self.p == 0:
            return x
        
        # x: (batch, step, hidden)
        m = torch.empty(x.size(0), x.size(1), 1, device=x.device).bernoulli(p=1-self.p)
        return x * m / (1 - self.p)
    
    def extra_repr(self):
        return f"p={self.p}"
    
    