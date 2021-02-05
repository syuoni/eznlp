# -*- coding: utf-8 -*-
import torch


def _is_text_like(text_like):
    if isinstance(text_like, str):
        return True
    elif isinstance(text_like, list):
        return all(isinstance(x, str) or _is_text_like(x) for x in text_like)
    else:
        return False
    
    
class TensorWrapper(object):
    def __init__(self, **kwargs):
        self.add_attributes(**kwargs)
        
    def add_attributes(self, **kwargs):
        for name, possible_attr in kwargs.items():
            # Do not register attributes with values of None
            if possible_attr is None:
                continue
            elif isinstance(possible_attr, (torch.Tensor, TensorWrapper)):
                pass
            # Exceptions for text-like attributes (as inputs to pretrained models)
            elif _is_text_like(possible_attr):
                pass
            elif isinstance(possible_attr, list):
                assert all(isinstance(sub_attr, (torch.Tensor, TensorWrapper)) for sub_attr in possible_attr)
            elif isinstance(possible_attr, dict):
                assert all(isinstance(sub_attr, (torch.Tensor, TensorWrapper)) for sub_attr in possible_attr.values())
            else:
                raise TypeError(f"Invalid input to `TensorWrapper`: {possible_attr}")    
            setattr(self, name, possible_attr)
            
            
    @property
    def device(self):
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, (torch.Tensor, TensorWrapper)):
                if attr.device is not None:
                    return attr.device
            elif isinstance(attr, list) and len(attr) > 0:
                attr0 = attr[0]
                if isinstance(attr0, (torch.Tensor, TensorWrapper)):
                    return attr0.device
            elif isinstance(attr, dict) and len(attr) > 0:
                attr0 = next(iter(attr.values()))
                if isinstance(attr0, (torch.Tensor, TensorWrapper)):
                    return attr0.device
        else:
            return None
        
        
    def _apply_to_tensors(self, func):
        """
        This function must return `self`.
        """
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, func(attr))
            elif isinstance(attr, TensorWrapper):
                setattr(self, attr_name, attr._apply_to_tensors(func))
            elif isinstance(attr, list) and len(attr) > 0:
                attr0 = attr[0]
                if isinstance(attr0, torch.Tensor):
                    setattr(self, attr_name, [func(x) for x in attr])
                elif isinstance(attr0, TensorWrapper):
                    setattr(self, attr_name, [x._apply_to_tensors(func) for x in attr])
            elif isinstance(attr, dict) and len(attr) > 0:
                attr0 = next(iter(attr.values()))
                if isinstance(attr0, torch.Tensor):
                    setattr(self, attr_name, {k: func(v) for k, v in attr.items()})
                elif isinstance(attr0, TensorWrapper):
                    setattr(self, attr_name, {k: v._apply_to_tensors(func) for k, v in attr.items()})
        return self
        
    def pin_memory(self):
        return self._apply_to_tensors(lambda x: x.pin_memory())
        
    def to(self, *args, **kwargs):
        return self._apply_to_tensors(lambda x: x.to(*args, **kwargs))
    
    
    
class Batch(TensorWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __repr__(self):
        return "Batch with attributes: {}".format(", ".join(self.__dict__))
        