# -*- coding: utf-8 -*-
from typing import Union, List
import logging
import subprocess
import torch
import numpy
import matplotlib

logger = logging.getLogger(__name__)


class LRLambda(object):
    @staticmethod
    def constant_lr():
        return lambda step: 1.0
        
    @staticmethod
    def constant_lr_with_warmup(num_warmup_steps: int):
        assert num_warmup_steps >= 1
        
        def lr_lambda(step: int):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            else:
                return 1.0
        return lr_lambda
        
    @staticmethod
    def linear_decay_lr_with_warmup(num_warmup_steps: int, num_total_steps: int):
        assert num_warmup_steps >= 1
        assert num_total_steps >= num_warmup_steps
        
        def lr_lambda(step: int):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            elif step < num_total_steps:
                return (num_total_steps - step) / (num_total_steps - num_warmup_steps)
            else:
                return 0.0
        return lr_lambda
        
    @staticmethod
    def exponential_decay_lr_with_warmup(num_warmup_steps: int, num_period_steps: int=None, gamma: float=0.9):
        if num_period_steps is None:
            num_period_steps = num_warmup_steps
        assert num_warmup_steps >= 1
        assert num_period_steps >= 1
        assert 0 < gamma < 1
        
        def lr_lambda(step: int):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            else:
                return gamma ** ((step - num_warmup_steps) / num_period_steps)
        return lr_lambda
        
    @staticmethod
    def power_decay_lr_with_warmup(num_warmup_steps: int, alpha: float=0.5):
        assert num_warmup_steps >= 1
        assert 0 < alpha < 1
        
        def lr_lambda(step: int):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            else:
                return (step / num_warmup_steps) ** (-alpha)
        return lr_lambda
        
    @staticmethod
    def plot_lr_lambda(lr_lambda, num_total_steps: int):
        x = numpy.arange(0, num_total_steps, num_total_steps//200)
        y = numpy.array([lr_lambda(xi) for xi in x])
        
        fig, ax = matplotlib.pyplot.subplots(figsize=(8, 3))
        ax.plot(x, y)
        matplotlib.pyplot.show()



def collect_params(model: torch.nn.Module, param_groups: list):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model
    param_groups : list
        [{'params': List[torch.nn.Parameter], 'lr': lr, ...}, ...]
    """
    existing = [params for group in param_groups for params in group['params']]
    missing = []
    for params in model.parameters():
        if all(params is not e_params for e_params in existing):
            missing.append(params)
    return missing


def check_param_groups(model: torch.nn.Module, param_groups: list, verbose=True):
    num_grouped_params = sum(count_params(group['params'], verbose=False) for group in param_groups)
    num_model_params = count_params(model, verbose=False)
    is_equal = (num_grouped_params == num_model_params)
    
    if verbose:
        if is_equal:
            logger.info(f"Grouped parameters ({num_grouped_params:,}) == Model parameters ({num_model_params:,})")
        else:
            logger.warning(f"Grouped parameters ({num_grouped_params:,}) != Model parameters ({num_model_params:,})")
    return is_equal


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
        logger.info(f"The model has {num_trainable + num_frozen:,} parameters, "
                    f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen.")
    
    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen


def auto_device(min_memory: int=2048):
    """Return the cuda device with the most free memory, if available; otherwise return the `cpu` device. 
    
    If torch's device order is inconsistent with that by nvidia-smi, use the following command before running the whole process: 
    ```bash
    $ export CUDA_DEVICE_ORDER=PCI_BUS_ID
    ```
    
    References
    ----------
    [1] https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow
    [2] https://discuss.pytorch.org/t/gpu-devices-nvidia-smi-and-cuda-get-device-name-output-appear-inconsistent/13150
    """
    logger.info("Automatically allocating device...")
    
    if not torch.cuda.is_available():
        logger.info("Cuda device is unavailable, device `cpu` returned")
        return torch.device('cpu')
    
    try:
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        free_memories = subprocess.check_output(COMMAND.split()).decode().strip().split('\n')[1:]
        free_memories = [int(x.split()[0]) for x in free_memories]
        assert len(free_memories) == torch.cuda.device_count()
    except:
        logger.warning("Cuda device information inquiry failed, device `cpu` returned")
        return torch.device('cpu')
    else:
        selected_id = numpy.argmax(free_memories)
        selected_mem = free_memories[selected_id]
        if selected_mem < min_memory:
            logger.warning(f"Cuda device `cuda:{selected_id}` with maximum free memory {selected_mem} MiB "
                           f"fails to meet the requirement {min_memory} MiB, device `cpu` returned")
            return torch.device('cpu')
        else:
            logger.info(f"Cuda device `cuda:{selected_id}` with free memory {selected_mem} MiB "
                        f"successfully allocated, device `cuda:{selected_id}` returned")
            return torch.device('cuda', selected_id)
