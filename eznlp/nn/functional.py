# -*- coding: utf-8 -*-
import torch


def seq_lens2mask(seq_lens: torch.LongTensor, max_len: int=None):
    """
    Parameters
    ----------
    seq_lens : torch.LongTensor (batch, )
    max_len : int, optional
    
    Returns
    -------
    mask : torch.BoolTensor
        The positions with values of True are MASKED, while the others are NOT MASKED. 
    """
    max_len = seq_lens.max().item() if max_len is None else max_len
    steps = torch.arange(max_len, device=seq_lens.device).repeat(seq_lens.size(0), 1)
    return (steps >= seq_lens.unsqueeze(1))


# TODO: Alternative aggregation method?
def aggregate_tensor_by_group(tensor: torch.FloatTensor, 
                              group_by: torch.LongTensor, 
                              agg_step: int=None):
    """
    Parameters
    ----------
    tensor : torch.FloatTensor (batch, ori_step, hidden)
        The tensor to be aggregate. 
    group_by : torch.LongTensor (batch, ori_step)
        The tensor indicating the positions after aggregation. 
        Positions being negative values are NOT used in aggregation. 
    """
    agg_step = (group_by.max().item() + 1) if agg_step is None else agg_step
    
    # pos_proj: (agg_step, ori_step)
    pos_proj = torch.arange(agg_step, device=group_by.device).unsqueeze(1).repeat(1, group_by.size(1))
    
    # pos_proj: (batch, agg_step, ori_step)
    pos_proj = torch.nn.functional.normalize((pos_proj.unsqueeze(0) == group_by.unsqueeze(1)).type(torch.float), p=1, dim=2)
    
    # agg_tensor: (batch, agg_step, hidden)
    return pos_proj.bmm(tensor)

