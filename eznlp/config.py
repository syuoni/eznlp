# -*- coding: utf-8 -*-
from typing import List, Mapping
from collections import OrderedDict
import torch


class Config(object):
    def __init__(self, **kwargs):
        for key, attr in kwargs.items():
            setattr(self, key, attr)
        
    @property
    def is_valid(self):
        for attr in self.__dict__.values():
            if isinstance(attr, (Config, ConfigDict)):
                if not attr.is_valid:
                    return False
            else:
                if attr is None:
                    return False
        return True
    
    def __repr__(self):
        kwargs_repr = ', '.join(f"{key}={attr}" for key, attr in self.__dict__.items())
        return f"{self.__class__.__name__}({kwargs_repr})"
    
    
class ConfigList(object):
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
            return sum(getattr(config, name) for config in self.config_list)
        elif name == 'arch':
            return '-'.join(getattr(config, name) for config in self.config_list)
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
    
    def __repr__(self):
        config_repr = "".join(f"\t{repr(config)}\n" for config in self.config_list)
        return f"{self.__class__.__name__}([\n{config_repr}])"
    
    
class ConfigDict(object):
    def __init__(self, config_dict: Mapping[str, Config]):
        # NOTE: `torch.nn.ModuleDict` is an **ordered** dictionary
        # NOTE: The order should be preserved. 
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
            return sum(getattr(config, name) for config in self.config_dict.values())
        elif name == 'arch':
            return '-'.join(getattr(config, name) for config in self.config_dict.values())
        else:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute {name}")
    
    def __repr__(self):
        config_repr = "".join(f"\t{key}={repr(config)}\n" for key, config in self.config_dict.items())
        return f"{self.__class__.__name__}([\n{config_repr}])"
    
    
class VocabConfig(Config):
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
    
    
    
class CharConfig(VocabConfig):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CNN')
        
        if self.arch.lower() not in ('lstm', 'gru', 'cnn'):
            raise ValueError(f"Invalid char-level architecture {self.arch}")
            
        if self.arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.dropout = kwargs.pop('dropout', 0.5)
        super().__init__(**kwargs)
        
        
class TokenConfig(VocabConfig):
    def __init__(self, **kwargs):
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.max_len = kwargs.pop('max_len', 300)
        self.use_pos_emb = kwargs.pop('use_pop_emb', False)
        
        self.freeze = kwargs.pop('freeze', False)
        self.scale_grad_by_freq = kwargs.pop('scale_grad_by_freq', False)
        super().__init__(**kwargs)
        
        
class EnumConfig(VocabConfig):
    def __init__(self, **kwargs):
        self.emb_dim = kwargs.pop('emb_dim', 25)
        super().__init__(**kwargs)
        
        
class ValConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.emb_dim = kwargs.pop('emb_dim', 25)
        super().__init__(**kwargs)
        
    def trans(self, values):
        return (torch.tensor(values, dtype=torch.float) * 2 - 1) / 10
        
        
class EmbedderConfig(Config):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        token: TokenConfig
        char: CharConfig
        enum: ConfigDict[str -> EnumConfig]
        val: ConfigDict[str -> ValConfig]
        """
        self.token = kwargs.pop('token', TokenConfig())
        self.char = kwargs.pop('char', None)
        self.enum = kwargs.pop('enum', None)
        self.val = kwargs.pop('val', None)
        super().__init__(**kwargs)
        
    @property
    def is_valid(self):
        if self.token is None or not self.token.is_valid:
            return False
        if self.char is not None and not self.char.is_valid:
            return False
        if self.enum is not None and not self.enum.is_valid:
            return False
        if self.val is not None and not self.val.is_valid:
            return False
        return True
        
    @property
    def out_dim(self):
        out_dim = 0
        out_dim += self.token.emb_dim if (self.token is not None) else 0
        out_dim += self.char.out_dim if (self.char is not None) else 0
        out_dim += self.enum.emb_dim if (self.enum is not None) else 0
        out_dim += self.val.emb_dim if (self.val is not None) else 0
        return out_dim
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"\ttoken={repr(self.token)}\n"
                f"\tchar ={repr(self.char)}\n"
                f"\tenum ={repr(self.enum)}\n"
                f"\tval  ={repr(self.val)})")
    
    
class EncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'LSTM')
        self.in_dim = kwargs.pop('in_dim', None)
        
        if self.arch.lower() == 'shortcut':
            self.hid_dim = kwargs.pop('hid_dim', None)
            self.dropout = kwargs.pop('dropout', 0.0)
        
        elif self.arch.lower() in ('lstm', 'gru'):
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.num_layers = kwargs.pop('num_layers', 1)
            self.dropout = kwargs.pop('dropout', 0.0)
            
        elif self.arch.lower() == 'cnn':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.kernel_size = kwargs.pop('kernel_size', 3)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.25)
            
        elif self.arch.lower() == 'transformer':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.nhead = kwargs.pop('nhead', 8)
            self.pf_dim = kwargs.pop('pf_dim', 256)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.1)
            
        else:
            raise ValueError(f"Invalid encoder architecture {self.arch}")
        
        super().__init__(**kwargs)
        
        
        
class PreTrainedEmbedderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'PTM')
        self.out_dim = kwargs.pop('out_dim')
        self.freeze = kwargs.pop('freeze', False)
        
        if self.arch.lower() == 'elmo':
            self.lstm_stateful = kwargs.pop('lstm_stateful', False)
        elif self.arch.lower() in ('bert', 'roberta', 'albert'):
            self.tokenizer = kwargs.pop('tokenizer')
        else:
            raise ValueError(f"Invalid pretrained embedder architecture {self.arch}")
        
        super().__init__(**kwargs)
        
        