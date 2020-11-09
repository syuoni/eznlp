# -*- coding: utf-8 -*-
import torch
from .functional import max_pooling, mean_pooling


class MaxPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        return max_pooling(x, mask)
    
    
class MeanPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        return mean_pooling(x, mask)
    

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
    
    