# -*- coding: utf-8 -*-
import torch

from ..init import reinit_layer_
from ..utils import _nonlinearity2activation


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop_rate: float):
        super().__init__()
        self.proj_layer = torch.nn.Linear(in_dim, out_dim)
        self.relu = torch.nn.ReLU()
        reinit_layer_(self.proj_layer, 'relu')
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x: torch.Tensor):
        return self.relu(self.proj_layer(self.dropout(x)))



class ConvBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, drop_rate: float, padding_mode: str='both', nonlinearity: str='relu'):
        super().__init__()
        padding_size = 0 if padding_mode.lower() == 'none' else kernel_size-1
        
        if nonlinearity.lower() == 'glu':
            self.conv = torch.nn.Conv1d(in_dim, out_dim*2, kernel_size=kernel_size, padding=padding_size)
            self.activation = torch.nn.GLU(dim=1)
        else:
            self.conv = torch.nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding_size)
            self.activation = _nonlinearity2activation(nonlinearity)
        
        reinit_layer_(self.conv, nonlinearity)
        self.dropout = torch.nn.Dropout(drop_rate)
        
        assert padding_mode.lower() in ('both', 'pre', 'post', 'none')
        self.padding_mode = padding_mode
        
    @property
    def kernel_size(self):
        return self.conv.kernel_size[0]
        
    @property
    def _pre_trim_size(self):
        if self.padding_mode.lower() == 'both':
            return (self.kernel_size - 1) // 2
        elif self.padding_mode.lower() == 'pre':
            # If paddings are in front, do not trim the front tensors; vice versa. 
            return 0
        else:
            return self.kernel_size - 1
        
    @property
    def _post_trim_size(self):
        return self.kernel_size - 1 - self._pre_trim_size
        
    def _trim(self, x: torch.Tensor):
        assert self.padding_mode.lower() in ('both', 'pre', 'post'), f"Illegal to trim with `padding_mode` {self.padding_mode}"
        assert x.dim() == 3
        # x: (batch, channels, step)
        return x[:, :, self._pre_trim_size:x.size(-1)-self._post_trim_size]
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        # NOTE: It would be better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layers have such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        
        # x: (batch, in_dim=channels, step)
        # mask: (batch, step)
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1), 0)
        
        # conved: (batch, out_dim=channels, step)
        conved = self.activation(self.conv(self.dropout(x)))
        if self.padding_mode.lower() == 'none':
            return conved
        else:
            return self._trim(conved)
