# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch
from torchtext.experimental.vocab import Vocab
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor


class Config(object):
    def __init__(self, **kwargs):
        for key, attr in kwargs.items():
            setattr(self, key, attr)
        
    @property
    def is_valid(self):
        for key, attr in self.__dict__.items():
            if isinstance(attr, (Config, ConfigDict)):
                if not attr.is_valid:
                    return False
            else:
                if attr is None:
                    return False
        return True
    
    def __repr__(self):
        kwargs_str = ', \n'.join(f"{key}={attr}" for key, attr in self.__dict__.items() \
                               if not (key == 'trans' or key == 'vocab' or key.startswith('idx2') or key.endswith('2idx')))
        return f"{self.__class__.__name__}({kwargs_str})"
    
    
class ConfigDict(object):
    def __init__(self, config_dict: OrderedDict):
        self.config_dict = config_dict
        
    @property
    def is_valid(self):
        for key, config in self.config_dict.items():
            if not config.is_valid:
                return False
        return True
    
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
            raise AttributeError(f"{type(self)} object has no attribute {name}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config_dict={repr(self.config_dict)})"
    
        
class CharEncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CNN')
        
        vocab = kwargs.pop('vocab', None)
        self.set_vocab(vocab)
        
        if self.arch.lower() not in ('lstm', 'gru', 'cnn'):
            raise ValueError(f"Invalid char-level architecture {self.arch}")
            
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.dropout = kwargs.pop('dropout', 0.5)
        
        if self.arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            
        super().__init__(**kwargs)
        
        
    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab
        self.trans = sequential_transforms(vocab_func(vocab), totensor(torch.long))
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
        
        
class TokenEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        vocab = kwargs.pop('vocab', None)
        self.set_vocab(vocab)
        
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.max_len = kwargs.pop('max_len', 300)
        self.use_pos_emb = kwargs.pop('use_pop_emb', False)
        super().__init__(**kwargs)
        
    def set_vocab(self, vocab: Vocab):
        # It is generally recommended to return cpu tensors in multi-process loading. 
        # See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        self.vocab = vocab
        self.trans = sequential_transforms(vocab_func(vocab), totensor(torch.long))
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
    
    
class EnumEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        vocab = kwargs.pop('vocab', None)
        self.set_vocab(vocab)
        
        self.emb_dim = kwargs.pop('emb_dim', 100)
        super().__init__(**kwargs)
        
    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab
        self.trans = sequential_transforms(vocab_func(vocab), totensor(torch.long))
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
        
    
        
class ValEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.trans = sequential_transforms(totensor(torch.float), lambda x: (x*2-1) / 10)
        super().__init__(**kwargs)
        
        
class EmbedderConfig(Config):
    def __init__(self, **kwargs):
        self.token = kwargs.pop('token', TokenEmbeddingConfig())
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
            self.dropout = kwargs.pop('dropout', 0.5)
            
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
        
        
class PreTrainedModelConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'PTM')
        self.hid_dim = kwargs.pop('hid_dim')
        self.tokenizer = kwargs.pop('tokenizer')
        
        super().__init__(**kwargs)
        