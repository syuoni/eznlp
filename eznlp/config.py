# -*- coding: utf-8 -*-
from typing import List, Mapping
from collections import OrderedDict
import torch
import torch.nn as nn


def _add_indents(config_str: str, num_spaces: int=2):
    lines = config_str.split('\n')
    if len(lines) == 1:
        return config_str
    else:
        lines = [lines[0]] + [' '*num_spaces + line for line in lines[1:]]
        return '\n'.join(lines)
    

class Config(object):
    def __init__(self, **kwargs):
        for key, attr in kwargs.items():
            setattr(self, key, attr)
        
    @property
    def is_valid(self):
        for attr in self.__dict__.values():
            if isinstance(attr, Config):
                if not attr.is_valid:
                    return False
            else:
                if attr is None:
                    return False
        return True
    
    def instantiate(self):
        raise NotImplementedError("Not Implemented `instantiate`")
        
    def __repr__(self):
        return self._repr_non_config_attrs(self.__dict__)
        
    def _repr_non_config_attrs(self, attr_dict: dict):
        main_str = self.__class__.__name__ + '('
        main_str += ', '.join(f"{key}={attr}" for key, attr in attr_dict.items())
        main_str += ')'
        return main_str
    
    def _repr_config_attrs(self, attr_dict: dict):
        main_str = self.__class__.__name__ + '(\n'
        main_str += '\n'.join(f"  ({key}): {_add_indents(repr(attr))}" for key, attr in attr_dict.items())
        main_str += '\n)'
        return main_str
    
    
class ConfigList(Config):
    def __init__(self, config_list: List[Config]):
        # NOTE: The order should be preserved. 
        if isinstance(config_list, list):
            assert all(isinstance(value, Config) for value in config_list)
            self.config_list = config_list
        else:
            raise ValueError(f"Invalid type of config_list {config_list}")
            
    @property
    def is_valid(self):
        for config in self.config_list:
            if not config.is_valid:
                return False
        return True
    
    def __len__(self):
        return len(self.config_list)
    
    def __iter__(self):
        return iter(self.config_list)
    
    def __getitem__(self, i):
        return self.config_list[i]
    
    def __getattr__(self, name):
        if name.endswith('_dim'):
            return sum(getattr(config, name) for config in self)
        elif name == 'arch':
            return '-'.join(getattr(config, name) for config in self)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
    
    def instantiate(self):
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        return nn.ModuleList([config.instantiate() for config in self])
    
    def __repr__(self):
        return self._repr_config_attrs({i: config for i, config in enumerate(self)})
    
    
class ConfigDict(Config):
    def __init__(self, config_dict: Mapping[str, Config]):
        # NOTE: `torch.nn.ModuleDict` is an **ordered** dictionary
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        if isinstance(config_dict, OrderedDict):
            assert all(isinstance(value, Config) for key, value in config_dict.items())
            self.config_dict = config_dict
        elif isinstance(config_dict, list):
            assert all(isinstance(value, Config) for key, value in config_dict)
            self.config_dict = OrderedDict(config_dict)
        else:
            raise ValueError(f"Invalid type of config_dict {config_dict}")
        
    @property
    def is_valid(self):
        for config in self.config_dict.values():
            if not config.is_valid:
                return False
        return True
    
    def __len__(self):
        return len(self.config_dict)
    
    def keys(self):
        return self.config_dict.keys()
    
    def values(self):
        return self.config_dict.values()
    
    def items(self):
        return self.config_dict.items()
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __getattr__(self, name):
        if name.endswith('_dim'):
            return sum(getattr(config, name) for config in self.values())
        elif name == 'arch':
            return '-'.join(getattr(config, name) for config in self.values())
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
    
    def instantiate(self):
        # NOTE: `torch.nn.ModuleDict` is an **ordered** dictionary
        # NOTE: The order should be preserved here. 
        return nn.ModuleDict([(key, config.instantiate()) for key, config in self.items()])
    
    def __repr__(self):
        return self._repr_config_attrs(self.config_dict)
    
    
class ConfigwithVocab(Config):
    def __init__(self, **kwargs):
        self.vocab = kwargs.pop('vocab', None)
        super().__init__(**kwargs)
        
    def trans(self, tok_iter):
        # It is generally recommended to return cpu tensors in multi-process loading. 
        # See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        return torch.tensor([self.vocab[tok] for tok in tok_iter], dtype=torch.long)
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
    
    
    