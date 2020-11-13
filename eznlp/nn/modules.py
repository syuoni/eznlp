# -*- coding: utf-8 -*-
import torch


class MaxPooling(torch.nn.Module):
    """
    Max Pooling over steps. 
    
    Parameters
    ----------
    x : torch.FloatTensor (batch, step, hidden)
    mask : torch.BoolTensor (batch, step)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return x_masked.max(dim=1).values
    
    
class MeanPooling(torch.nn.Module):
    """
    Mean Pooling over steps. 
    
    Parameters
    ----------
    x : torch.FloatTensor (batch, step, hidden)
    mask : torch.BoolTensor (batch, step)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
        seq_lens = mask.size(1) - mask.sum(dim=1)
        return x_masked.sum(dim=1) / seq_lens.unsqueeze(1)
    
    

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
    
    