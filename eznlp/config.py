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
    
    _name_sep = '-'
    
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            logger.warning(f"Some configurations are set without checking: {kwargs}, "
                           "which may be never used.")
            for key, attr in kwargs.items():
                setattr(self, key, attr)
                
    @property
    def valid(self):
        for name, attr in self.__dict__.items():
            if attr is None:
                return False
            elif isinstance(attr, Config) and not attr.valid:
                return False
        return True
        
    @property
    def name(self):
        raise NotImplementedError("Not Implemented `name`")
        
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
    def __init__(self, config_list: List[Config]=None):
        if config_list is None:
            config_list = []
        elif not isinstance(config_list, list):
            config_list = list(config_list)
            
        assert all(isinstance(c, Config) for c in config_list)
        self.config_list = config_list
        
        
    @property
    def valid(self):
        return len(self.config_list) > 0 and all(c.valid for c in self.config_list)
    
    @property
    def name(self):
        return self._name_sep.join(c.name for c in self.config_list)
        
    def __len__(self):
        return len(self.config_list)
    
    def __iter__(self):
        return iter(self.config_list)
    
    def __getitem__(self, i):
        return self.config_list[i]
    
    def __setitem__(self, i, c: Config):
        assert isinstance(c, Config)
        self.config_list[i] = c
    
    def append(self, c: Config):
        assert isinstance(c, Config)
        self.config_list.append(c)
    
    @property
    def out_dim(self):
        return sum(c.out_dim for c in self.config_list)
        
    def instantiate(self):
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        return torch.nn.ModuleList([c.instantiate() for c in self.config_list])
        
    def __repr__(self):
        return self._repr_config_attrs({i: c for i, c in enumerate(self.config_list)})
        
        
class ConfigDict(Config):
    def __init__(self, config_dict: Mapping[str, Config]=None):
        if config_dict is None:
            config_dict = {}
        # NOTE: `torch.nn.ModuleDict` is an **ordered** dictionary
        elif not isinstance(config_dict, OrderedDict):
            config_dict = OrderedDict(config_dict)
            
        assert all(isinstance(c, Config) for c in config_dict.values())
        self.config_dict = config_dict
        
        
    @property
    def valid(self):
        return len(self.config_dict) > 0 and all(c.valid for c in self.config_dict.values())
        
    @property
    def name(self):
        return self._name_sep.join(c.name for c in self.config_dict.values())
    
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
    
    def __setitem__(self, key, c: Config):
        assert isinstance(c, Config)
        self.config_dict[key] = c
    
    @property
    def out_dim(self):
        return sum(c.out_dim for c in self.config_dict.values())
        
    def instantiate(self):
        # NOTE: The order should be consistent here and in the corresponding `forward`. 
        return torch.nn.ModuleDict([(k, c.instantiate()) for k, c in self.config_dict.items()])
        
    def __repr__(self):
        return self._repr_config_attrs(self.config_dict)
