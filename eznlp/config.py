# -*- coding: utf-8 -*-

class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        
class EncoderConfig(Config):
    def __init__(self, arch: str, **kwargs):
        self.arch = arch
        
        if arch.lower() == 'shortcut':
            assert 'hid_dim' in kwargs
            self.hid_dim = kwargs.pop('hid_dim')
        
        elif arch.lower() in ('lstm', 'gru'):
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.num_layers = kwargs.pop('num_layers', 1)
            self.dropout = kwargs.pop('dropout', 0.5)
            
        elif arch.lower() == 'cnn':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.kernel_size = kwargs.pop('kernel_size', 3)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.25)
            
        elif arch.lower() == 'transformer':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.nhead = kwargs.pop('nhead', 8)
            self.pf_dim = kwargs.pop('pf_dim', 256)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.1)
            
        else:
            raise ValueError(f"Invalid encoder architecture {arch}")
        
        super().__init__(**kwargs)
        
        
class CharEncoderConfig(Config):
    def __init__(self, arch: str, voc_dim: int, pad_idx: int, **kwargs):
        self.arch = arch
        self.voc_dim = voc_dim
        self.pad_idx = pad_idx
        
        if arch.lower() not in ('lstm', 'gru', 'cnn'):
            raise ValueError(f"Invalid char-level architecture {arch}")
            
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.dropout = kwargs.pop('dropout', 0.5)
        
        if arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            
        super().__init__(**kwargs)
        
        
class TokenEmbeddingConfig(Config):
    def __init__(self, **kwargs):
        
        
        super().__init__(**kwargs)
        
        
        
class EmbedderConfig(Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        