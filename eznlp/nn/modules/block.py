# -*- coding: utf-8 -*-
import torch

from ..init import reinit_layer_


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
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, drop_rate: float):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.relu = torch.nn.ReLU()
        reinit_layer_(self.conv, 'relu')
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        # NOTE: It would be better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layers have such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        
        # x: (batch, in_dim=channels, step)
        # mask: (batch, step)
        x.masked_fill_(mask.unsqueeze(1), 0)
        return self.relu(self.conv(self.dropout(x)))



class GehringConvBlock(torch.nn.Module):
    def __init__(self, hid_dim: int, kernel_size: int, drop_rate: float, scale: float=0.5**0.5):
        super().__init__()
        self.conv = torch.nn.Conv1d(hid_dim, hid_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.glu = torch.nn.GLU(dim=1)
        reinit_layer_(self.conv, 'sigmoid')
        self.dropout = torch.nn.Dropout(drop_rate)
        self.scale = scale
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        # x: (batch, hid_dim=channels, step)
        # mask: (batch, step)
        x.masked_fill_(mask.unsqueeze(1), 0)
        conved = self.glu(self.conv(self.dropout(x)))
        
        # Residual connection
        return (x + conved) * self.scale
