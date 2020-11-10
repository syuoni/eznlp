# -*- coding: utf-8 -*-
import torch
from torchtext.experimental.vectors import Vectors

from ..dataset_utils import Batch
from ..nn.init import reinit_embedding_, reinit_layer_
from ..config import Config, ConfigwithVocab


class EnumConfig(ConfigwithVocab):
    def __init__(self, **kwargs):
        self.emb_dim = kwargs.pop('emb_dim', 25)
        super().__init__(**kwargs)
        
    def instantiate(self):
        return EnumEmbedding(self)
    
    
class EnumEmbedding(torch.nn.Module):
    def __init__(self, config: EnumConfig):
        super().__init__()
        self.emb = torch.nn.Embedding(config.voc_dim, config.emb_dim, padding_idx=config.pad_idx)
        reinit_embedding_(self.emb)
        
    def forward(self, enum_ids: torch.Tensor):
        return self.emb(enum_ids)
    

class ValConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.emb_dim = kwargs.pop('emb_dim', 25)
        super().__init__(**kwargs)
        
    def trans(self, values):
        return (torch.tensor(values, dtype=torch.float) * 2 - 1) / 10
    
    def instantiate(self):
        return ValEmbedding(self)
    
    
class ValEmbedding(torch.nn.Module):
    def __init__(self, config: ValConfig):
        super().__init__()
        # NOTE: Two reasons why this layer does not have activation. 
        # (1) Activation function should been applied after batch-norm / layer-norm. 
        # (2) This layer is semantically an embedding layer, which typically does NOT require activation. 
        self.proj = torch.nn.Linear(config.in_dim, config.emb_dim)
        reinit_layer_(self.proj, 'linear')
        
    def forward(self, val_ins: torch.Tensor):
        return self.proj(val_ins)
    

class TokenConfig(ConfigwithVocab):
    def __init__(self, **kwargs):
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.max_len = kwargs.pop('max_len', 300)
        self.use_pos_emb = kwargs.pop('use_pop_emb', False)
        
        self.freeze = kwargs.pop('freeze', False)
        self.scale_grad_by_freq = kwargs.pop('scale_grad_by_freq', False)
        super().__init__(**kwargs)
    
    def instantiate(self, pretrained_vectors: Vectors=None):
        return TokenEmbedding(self, pretrained_vectors=pretrained_vectors)
    

class TokenEmbedding(torch.nn.Module):
    def __init__(self, config: TokenConfig, pretrained_vectors: Vectors=None):
        super().__init__()
        self.word_emb = torch.nn.Embedding(config.voc_dim, config.emb_dim, padding_idx=config.pad_idx, 
                                           scale_grad_by_freq=config.scale_grad_by_freq)
        reinit_embedding_(self.word_emb, itos=config.vocab.get_itos(), pretrained_vectors=pretrained_vectors)
        self.freeze = config.freeze
        
        if config.use_pos_emb:
            self.pos_emb = torch.nn.Embedding(config.max_len, config.emb_dim)
            reinit_embedding_(self.pos_emb)
        
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        self.word_emb.requires_grad_(not self._freeze)
        
    def forward(self, tok_ids: torch.Tensor):
        # word_embedded: (batch, step, emb_dim)
        word_embedded = self.word_emb(tok_ids)
        
        # pos_embedded: (batch, step, emb_dim)
        if hasattr(self, 'pos_emb'):
            pos = torch.arange(word_embedded.size(1), device=word_embedded.device).repeat(word_embedded.size(0), 1)
            pos_embedded = self.pos_emb(pos)
            return (word_embedded + pos_embedded) * (0.5**0.5)
        else:
            return word_embedded



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
    
    def instantiate(self, pretrained_vectors: Vectors=None):
        return Embedder(self, pretrained_vectors)
    
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
    
    
class Embedder(torch.nn.Module):
    """
    `Embedder` forwards from inputs to embeddings. 
    """
    def __init__(self, config: EmbedderConfig, pretrained_vectors: Vectors=None):
        super().__init__()
        self.token_emb = config.token.instantiate(pretrained_vectors=pretrained_vectors)
        
        if config.char is not None:
            self.char_encoder = config.char.instantiate()
            
        if config.enum is not None:
            self.enum_embs = config.enum.instantiate()
                
        if config.val is not None:
            self.val_embs = config.val.instantiate()
            
            
    def get_token_embedded(self, batch: Batch):
        return self.token_emb(batch.tok_ids)
    
    def get_char_embedded(self, batch: Batch):
        return self.char_encoder(batch.char_ids, batch.tok_lens, batch.char_mask, batch.seq_lens)
    
    def get_enum_embedded(self, batch: Batch):
        return torch.cat([self.enum_embs[f](batch.enum[f]) for f in self.enum_embs], dim=-1)
        
    def get_val_embedded(self, batch: Batch):
        return torch.cat([self.val_embs[f](batch.val[f]) for f in self.val_embs], dim=-1)
    
    def forward(self, batch: Batch):
        embedded = []
        embedded.append(self.get_token_embedded(batch))
        
        if hasattr(self, 'char_encoder'):
            embedded.append(self.get_char_embedded(batch))
        if hasattr(self, 'enum_embs'):
            embedded.append(self.get_enum_embedded(batch))
        if hasattr(self, 'val_embs'):
            embedded.append(self.get_val_embedded(batch))
            
        return torch.cat(embedded, dim=-1)
    
    
    