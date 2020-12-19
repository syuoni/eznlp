# -*- coding: utf-8 -*-
from typing import Union, List
from collections import defaultdict
import torch


def count_params(model_or_params: Union[torch.nn.Module, torch.nn.Parameter, List[torch.nn.Parameter]], 
                 return_trainable=True, verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been iterated ONCE. 
    Hence, `model_or_params` passed-in must be a `List` of parameters (which can be iterated multiple times). 
    """
    if isinstance(model_or_params, torch.nn.Module):
        model_or_params = list(model_or_params.parameters())
    elif isinstance(model_or_params, torch.nn.Parameter):
        model_or_params = [model_or_params]
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of `torch.nn.Parameter`, "
                        "`model_or_params` should NOT be a `Generator`. ")
        
    num_trainable = sum(p.numel() for p in model_or_params if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_or_params if not p.requires_grad)
    if verbose:
        print(f"The model has {num_trainable + num_frozen:,} parameters, "
              f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen. ")
        
    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen


def build_param_groups_with_keyword2lr(model: torch.nn.Module, keyword2lr: dict, verbose=True):
    keyword2params = defaultdict(list)
    for name, params in model.named_parameters():
        for keyword in keyword2lr:
            if name.startswith(keyword):
                keyword2params[keyword].append(params)
                break
        else:
            keyword2params['<default>'].append(params)
            
    param_groups = []
    lr_lambdas = []
    for keyword, params in keyword2params.items():
        param_groups.append({'params': params})
        lr_lambdas.append(keyword2lr[keyword])
        
    if verbose:
        print(f"{len(param_groups)} parameter groups have been built")
        for keyword, params in keyword2params.items():
            print(f"Keyword: {keyword} | Parameters: {len(params)}")
        
    return param_groups, lr_lambdas


def check_param_groups_no_missing(param_groups: list, model: torch.nn.Module, verbose=True):
    num_grouped_params = sum(count_params(group['params'], verbose=False) for group in param_groups)
    num_model_params = count_params(model, verbose=False)
    is_equal = (num_grouped_params == num_model_params)
    if verbose:
        print(f"Grouped parameters ({num_grouped_params:,}) == Model parameters ({num_model_params:,})? {is_equal}")
    
    return is_equal

