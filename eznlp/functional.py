# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn.functional as F


# TODO: Alternative aggregation method?
def aggregate_tensor_by_group(tensor: Tensor, group_by: Tensor, agg_step: int=None):
    """
    Parameters
    ----------
    tensor : Tensor (batch, ori_step, hidden)
        The tensor to be aggregate. 
    group_by : Tesnor (batch, ori_step)
        The tensor indicating the positions after aggregation. 
        Positions being negative values are NOT used in aggregation. 
    """
    agg_step = (group_by.max().item() + 1) if agg_step is None else agg_step
    
    # pos_proj: (agg_step, ori_step)
    pos_proj = torch.arange(agg_step, device=group_by.device).unsqueeze(1).repeat(1, group_by.size(1))
    
    # pos_proj: (batch, agg_step, ori_step)
    pos_proj = F.normalize((pos_proj.unsqueeze(0) == group_by.unsqueeze(1)).type(torch.float), p=1, dim=2)
    
    # agg_tensor: (batch, agg_step, hidden)
    return pos_proj.bmm(tensor)
    
