# -*- coding: utf-8 -*-

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
    
    
class ConfigDict(object):
    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        
    @property
    def is_valid(self):
        for key, config in self.config_dict.items():
            if not config.is_valid:
                return False
        return True
    
    def __getattr__(self, name):
        if name.endsiwth('_dim'):
            return sum(getattr(config, name) for config in self.config_dict.values())
        elif name == 'arch':
            return '-'.join(getattr(config, name) for config in self.config_dict.values())
        else:
            raise AttributeError(f"{type(self)} object has no attribute {name}")
    
        
class CharEncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CNN')
        self.voc_dim = kwargs.pop('voc_dim', None)
        self.pad_idx = kwargs.pop('pad_idx', None)
        
        if self.arch.lower() not in ('lstm', 'gru', 'cnn'):
            raise ValueError(f"Invalid char-level architecture {self.arch}")
            
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.dropout = kwargs.pop('dropout', 0.5)
        
        if self.arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            
        super().__init__(**kwargs)
        
        
        
class TokenEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        self.vocab = kwargs.pop('vocab', None)
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.max_len = kwargs.pop('max_len', 300)
        self.use_pos_emb = kwargs.pop('use_pop_emb', False)
        super().__init__(**kwargs)
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pax_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
    
    
class EnumEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        self.vocab = kwargs.pop('vocab', None)
        self.emb_dim = kwargs.pop('emb_dim', 100)
        super().__init__(**kwargs)
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pax_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
        
    
        
class ValEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.emb_dim = kwargs.pop('emb_dim', 25)
        super().__init__(**kwargs)
        
    
        
class EmbedderConfig(Config):
    def __init__(self, **kwargs):
        self.token = kwargs.pop('token', None)
        self.char = kwargs.pop('char', None)
        self.enum = kwargs.pop('enum', None)
        self.val = kwargs.pop('val', None)
        super().__init__(**kwargs)
        
    @property
    def is_valid(self):
        return any([self.token, self.char, self.enum, self.val])
        
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
            assert 'hid_dim' in kwargs
            self.hid_dim = kwargs.pop('hid_dim')
        
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
        