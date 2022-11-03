# -*- coding: utf-8 -*-
import torch

def _create_is_like(criterion):
    def _is_like(x):
        if criterion(x):
            return True
        elif isinstance(x, list):
            return all(criterion(xi) or _is_like(xi) for xi in x)
        elif isinstance(x, dict):
            return all(criterion(xi) or _is_like(xi) for xi in x.values())
        else:
            return False
    return _is_like


def _create_apply(criterion, func):
    def _apply(x):
        if criterion(x):
            return func(x)
        elif isinstance(x, list):
            return [_apply(xi) for xi in x]
        elif isinstance(x, dict):
            return {k: _apply(xi) for k, xi in x.items()}
        else:
            return x
    return _apply


_is_string_like = _create_is_like(lambda x: isinstance(x, str))
_is_tensor_like = _create_is_like(lambda x: isinstance(x, (torch.Tensor, TensorWrapper)))



class TensorWrapper(object):
    """A wrapper of tensors.     
    """
    def __init__(self, **kwargs):
        self.add_attributes(**kwargs)
        
    def add_attributes(self, **kwargs):
        for name, possible_attr in kwargs.items():
            if possible_attr is None:
                # Do not register attributes with value of None
                continue
            elif _is_tensor_like(possible_attr) or _is_string_like(possible_attr):
                setattr(self, name, possible_attr)
            else:
                raise TypeError(f"Invalid input to `TensorWrapper`: {possible_attr}")
        
        
    def _apply_to_tensors(self, func):
        """Apply `func` to all tensors registered in this `TensorWrapper`. 
        
        Parameters
        ----------
        func: Callable
            a function appliable to `torch.Tensor`
        
        Notes
        -----
        `_adaptive_func` is a version appliable to both `torch.Tensor` and `TensorWrapper`. 
        `_apply` is a version *recursively* appliable to list/dict of `torch.Tensor` and `TensorWrapper`. 
        
        This function must return `self`.
        """
        def _adaptive_func(x):
            if isinstance(x, torch.Tensor):
                return func(x)
            else:
                return x._apply_to_tensors(func)
        
        _apply = _create_apply(lambda x: isinstance(x, (torch.Tensor, TensorWrapper)), _adaptive_func)
        
        for name, attr in self.__dict__.items():
            if _is_tensor_like(attr):
                setattr(self, name, _apply(attr))
        
        return self
        
    def pin_memory(self):
        return self._apply_to_tensors(lambda x: x.pin_memory())
        
    def to(self, *args, **kwargs):
        return self._apply_to_tensors(lambda x: x.to(*args, **kwargs))
        
    def cuda(self, *args, **kwargs):
        return self._apply_to_tensors(lambda x: x.cuda(*args, **kwargs))



class Batch(TensorWrapper):
    """A wrapper of batch. 
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __repr__(self):
        return "Batch with attributes: {}".format(", ".join(self.__dict__))



class TargetWrapper(TensorWrapper):
    """
    A wrapper of modeling targets (tensors) with underlying ground truth (e.g., labels, chunks, relations). 
    
    Notes
    -----
    (1) `training` is a flag adapting the contents of the current object. 
        If `training` is False, the object cannot expose the ground truth to **attributes that will be used in decoding**; 
        in other words, those **attributes that will be used in decoding** should be identical with or without the ground truth. 
        However, some **attributes that will not be used in decoding** may contain information of the ground truth for computing evaluation loss. 
    (2) Do NOT check the attributes (being tensors or not) for a target object. 
    """
    def __init__(self, training: bool=True):
        self.training = training
