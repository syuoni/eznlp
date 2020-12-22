# -*- coding: utf-8 -*-
import torch
from ..nn.init import reinit_layer_


class SequenceAttention(torch.nn.Module):
    """
    Attention over steps. 
    
    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, step)
    """
    def __init__(self, hid_dim: int, query_dim: int=None):
        if query_dim is None:
            query_dim = hid_dim
        self.proj_layer = torch.nn.Linear(hid_dim, query_dim)
        self.query = torch.nn.Parameter(torch.empty(query_dim))
        
        reinit_layer_(self.proj_layer, 'linear')
        uniform_range = (3 / query_dim) ** 0.5
        torch.nn.init.uniform_(self.query.data, -uniform_range, uniform_range)
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        # x: (batch, step, hid_dim) -> (batch, step, query_dim)
        x_projed = self.proj_layer(x)
        
        # energy/atten_weight: (batch, step)
        energy = x_projed.matmul(self.query)
        energy_masked = energy.masked_fill(mask, float('-inf'))
        atten_weight = torch.nn.functional.softmax(energy_masked, dim=-1)
        
        # atten_values: (batch, hid_dim)
        atten_values = atten_weight.unsqueeze(1).bmm(x).squeeze(1)
        return atten_values
        
    