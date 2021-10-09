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
        
    def forward(self, x: torch.FloatTensor):
        return self.relu(self.proj_layer(self.dropout(x)))



class ConvBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, drop_rate: float, nonlinearity: str='relu'):
        super().__init__()
        if nonlinearity.lower() == 'glu':
            self.conv = torch.nn.Conv1d(in_dim, out_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.activation = torch.nn.GLU(dim=1)
        else:
            self.conv = torch.nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
            self.activation = _nonlinearity2activation(nonlinearity)
        
        reinit_layer_(self.conv, nonlinearity)
        self.dropout = torch.nn.Dropout(drop_rate)
        
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor=None):
        # NOTE: It would be better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layers have such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        
        # x: (batch, in_dim=channels, step)
        # mask: (batch, step)
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1), 0)
        
        return self.activation(self.conv(self.dropout(x)))
