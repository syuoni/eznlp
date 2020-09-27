# -*- coding: utf-8 -*-
from typing import Union
from collections import defaultdict
import torch.nn as nn

def count_trainable_params(model_or_params: Union[nn.Module, list], verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been 
    iterated ONCE. Hence, `model_or_params` passed-in must be a `List` of 
    parameters (which can be iterated multiple times). 
    """
    if isinstance(model_or_params, nn.Module):
        model_or_params = model_or_params.parameters()
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of parameters, "
                        "`model_or_params` should NOT be a `Generator`. ")
        
    num_params = sum(p.numel() for p in model_or_params if p.requires_grad)
    if verbose:
        print(f"The model has {num_params:,} trainable parameters")
    return num_params


def build_param_groups_with_keyword2lr(model: nn.Module, keyword2lr: dict):
    keyword2params = defaultdict(list)
    for name, params in model.named_parameters():
        for keyword in keyword2lr:
            if keyword in name:
                keyword2params[keyword].append(params)
                break
        else:
            keyword2params['<default>'].append(params)
            
    param_groups = []
    lr_lambdas = []
    for keyword, params in keyword2params.items():
        param_groups.append({'params': params})
        lr_lambdas.append(keyword2lr[keyword])
        
    return param_groups, lr_lambdas


def check_param_groups_no_missing(param_groups: list, model: nn.Module, verbose=True):
    num_grouped_params = sum(count_trainable_params(group['params'], verbose=False) for group in param_groups)
    num_model_params = count_trainable_params(model, verbose=False)
    is_equal = (num_grouped_params == num_model_params)
    if verbose:
        print(f"Grouped parameters ({num_grouped_params:,}) == Model parameters ({num_model_params:,})? {is_equal}")
    
    return is_equal

