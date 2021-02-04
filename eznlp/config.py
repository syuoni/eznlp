# -*- coding: utf-8 -*-
from typing import List, Mapping
from collections import OrderedDict
import logging
import torch

logger = logging.getLogger(__name__)


def _add_indents(config_str: str, num_spaces: int=2):
    lines = config_str.split('\n')
    if len(lines) == 1:
        return config_str
    else:
        lines = [lines[0]] + [' '*num_spaces + line for line in lines[1:]]
        return '\n'.join(lines)
    

class Config(object):
    """
    `Config` stores and validates configurations of a model or assembly. 
    `Config` defines the methods to process data and tensors. 
    `Config.instantiate` is the suggested way to instantiate the model or assembly. 
    
    `Config` are NOT suggested to be registered as attribute of the corresponding model or assembly. 
    """
    
    _cache_attr_names = ['vectors', 'elmo', 'bert_like', 'flair_lm']
    
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            logger.warning(f"Some configurations are set without checking: {kwargs}, "
                           "which may be never used.")
            for key, attr in kwargs.items():
                setattr(self, key, attr)
                
    @property
    def valid(self):
        for name, attr in self.__dict__.items():
            if name in self._cache_attr_names:
                continue
            elif isinstance(attr, Config):
                if not attr.valid:
                    return False
            else:
                if attr is None:
                    return False
        return True
        
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
        
    def instantiate(self):
        raise NotImplementedError("Not Implemented `instantiate`")
    
    
class ConfigList(Config):
    def __init__(self, config_list: List[Config]):
        if not isinstance(config_list, list):
            config_list = list(config_list)
            
        assert all(isinstance(c, Config) for c in config_list)
        self.config_list = config_list
        
        
    @property
    def valid(self):
        return all(c.valid for c in self.config_list)
    
    def __len__(self):
        return len(self.config_list)
    
    def __iter__(self):
        return iter(self.config_list)
    
    def __getitem__(self, i):
        return self.config_list[i]
    
    @property
    def out_dim(self):
        return sum(c.out_dim for c in self.config_list)
        
    def instantiate(self):
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        return torch.nn.ModuleList([c.instantiate() for c in self.config_list])
        
    def __repr__(self):
        return self._repr_config_attrs({i: c for i, c in enumerate(self.config_list)})
        
        
class ConfigDict(Config):
    def __init__(self, config_dict: Mapping[str, Config]):
        # NOTE: `torch.nn.ModuleDict` is an **ordered** dictionary
        if not isinstance(config_dict, OrderedDict):
            config_dict = OrderedDict(config_dict)
            
        assert all(isinstance(c, Config) for c in config_dict.values())
        self.config_dict = config_dict
        
        
    @property
    def valid(self):
        return all(c.valid for c in self.config_dict.values())
        
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
    
    @property
    def out_dim(self):
        return sum(c.out_dim for c in self.config_dict.values())
        
    def instantiate(self):
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        return torch.nn.ModuleDict([(k, c.instantiate()) for k, c in self.config_dict.items()])
        
    def __repr__(self):
        return self._repr_config_attrs(self.config_dict)
    
    