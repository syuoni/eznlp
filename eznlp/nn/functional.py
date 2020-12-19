# -*- coding: utf-8 -*-
import torch


def seq_lens2mask(seq_lens: torch.LongTensor, max_len: int=None):
    """
    Convert `seq_lens` to `mask`. 
    
    Parameters
    ----------
    seq_lens : torch.LongTensor (batch, )
    max_len : int, optional
    
    Returns
    -------
    mask : torch.BoolTensor (batch, step)
        The positions with values of True are MASKED, while the others are NOT MASKED. 
    """
    max_len = seq_lens.max().item() if max_len is None else max_len
    steps = torch.arange(max_len, device=seq_lens.device).repeat(seq_lens.size(0), 1)
    return (steps >= seq_lens.unsqueeze(1))



def mask2seq_lens(mask: torch.BoolTensor):
    """
    Convert `mask` to `seq_lens`. 
    """
    return mask.size(1) - mask.sum(dim=1)



def sequence_pooling(x: torch.FloatTensor, mask: torch.BoolTensor, mode: str='mean'):
    """
    Pooling values over steps. 
    
    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, hid_dim)
    mode: str
        'mean', 'max', 'min'
    """
    if mode.lower() == 'mean':
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0)
        seq_lens = mask2seq_lens(mask)
        return x_masked.sum(dim=1) / seq_lens.unsqueeze(1)
    elif mode.lower() == 'max':
        x_masked = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return x_masked.max(dim=1).values
    elif mode.lower() == 'min':
        x_masked = x.masked_fill(mask.unsqueeze(-1), float('inf'))
        return x_masked.min(dim=1).values
    else:
        raise ValueError(f"Invalid pooling mode {mode}")
        
        
        
def sequence_group_aggregating(x: torch.FloatTensor, group_by: torch.LongTensor, agg_mode: str='mean', agg_step: int=None):
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
    if agg_mode.lower() not in ('mean', 'max', 'min', 'first', 'last'):
        raise ValueError(f"Invalid aggregation mode {agg_mode}")
    
    agg_step = (group_by.max().item() + 1) if agg_step is None else agg_step
    
    # pos_proj: (agg_step, ori_step)
    pos_proj = torch.arange(agg_step, device=group_by.device).unsqueeze(1).repeat(1, group_by.size(1))
    
    # pos_proj: (batch, agg_step, ori_step)
    pos_proj = (pos_proj.unsqueeze(0) == group_by.unsqueeze(1))
    
    if agg_mode.lower() in ('mean', 'first', 'last'):
        pos_proj_weight = _make_pos_proj_weight(pos_proj, agg_mode=agg_mode)
        
        # agg_tensor: (batch, agg_step, hidden)
        return pos_proj_weight.bmm(x)
    
    else:
        return _execute_pos_proj(x, pos_proj, agg_mode=agg_mode)
    
    
def _make_pos_proj_weight(pos_proj: torch.BoolTensor, agg_mode='mean'):
    if agg_mode.lower() == 'mean':
        return torch.nn.functional.normalize(pos_proj.type(torch.float), p=1, dim=2)
    elif agg_mode.lower() == 'first':
        pos_proj_weight = pos_proj & (pos_proj.cumsum(dim=-1) == 1)
        return pos_proj_weight.type(torch.float)
    elif agg_mode.lower() == 'last':
        pos_proj_weight = pos_proj & (pos_proj.cumsum(dim=-1) == pos_proj.sum(dim=-1, keepdim=True))
        return pos_proj_weight.type(torch.float)
    
    
def _execute_pos_proj(x: torch.FloatTensor, pos_proj: torch.BoolTensor, agg_mode='max'):
    proj_values = []
    for k in range(pos_proj.size(0)):
        curr_proj_values = []
        for curr_pos_proj in pos_proj[k]:
            if curr_pos_proj.sum() == 0:
                curr_proj_values.append(torch.zeros(x.size(-1)))
            elif agg_mode.lower() == 'max':
                curr_proj_values.append(x[k, curr_pos_proj].max(dim=0).values)
            elif agg_mode.lower() == 'min':
                curr_proj_values.append(x[k, curr_pos_proj].min(dim=0).values)
        proj_values.append(torch.stack(curr_proj_values))
    return torch.stack(proj_values)
            
            