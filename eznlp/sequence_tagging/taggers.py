# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
from torchtext.experimental.vectors import Vectors
from transformers import PreTrainedModel
from allennlp.modules.elmo import Elmo

from ..token import Token
from ..datasets_utils import Batch
from ..config import Config, ConfigList
from ..embedders import EmbedderConfig
from ..encoders import EncoderConfig
from ..pretrained_embedders import PreTrainedEmbedderConfig
from .decoders import DecoderConfig


class TaggerConfig(Config):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        embedder: EmbedderConfig
        encoders: ConfigList[EncoderConfig]
        elmo_embedder: PreTrainedEmbedderConfig
        bert_like_embedder: PreTrainedEmbedderConfig
        decoder: DecoderConfig
        """
        self.embedder = kwargs.pop('embedder', EmbedderConfig())
        self.encoders = kwargs.pop('encoders', ConfigList([EncoderConfig(arch='LSTM')]))
        
        self.elmo_embedder = kwargs.pop('elmo_embedder', None)
        self.bert_like_embedder = kwargs.pop('bert_like_embedder', None)
        
        self.intermediate = kwargs.pop('intermediate', None)
        self.decoder = kwargs.pop('decoder', DecoderConfig(arch='CRF'))
        super().__init__(**kwargs)
        
        
    @property
    def is_valid(self):
        if self.decoder is None or not self.decoder.is_valid:
            return False
        if self.embedder is None or not self.embedder.is_valid:
            return False
        
        if self.encoders is not None and self.encoders.is_valid:
            return True
        if self.elmo_embedder is not None and self.elmo_embedder.is_valid:
            return True
        if self.bert_like_embedder is not None and self.bert_like_embedder.is_valid:
            return True
        
        return False
        
    
    def _update_dims(self, ex_token: Token=None):
        if self.embedder.val is not None and ex_token is not None:
            for f, val_config in self.embedder.val.items():
                val_config.in_dim = getattr(ex_token, f).shape[0]
                
        if self.encoders is not None:
            for enc_config in self.encoders:
                enc_config.in_dim = self.embedder.out_dim
                if enc_config.arch.lower() == 'shortcut':
                    enc_config.hid_dim = self.embedder.out_dim
        
        full_hid_dim = 0
        full_hid_dim += self.encoders.hid_dim if self.encoders is not None else 0
        full_hid_dim += self.elmo_embedder.out_dim if self.elmo_embedder is not None else 0
        full_hid_dim += self.bert_like_embedder.out_dim if self.bert_like_embedder is not None else 0
        
        if self.intermediate is None:
            self.decoder.in_dim = full_hid_dim
        else:
            self.intermediate.in_dim = full_hid_dim
            self.decoder.in_dim = self.intermediate.hid_dim
            
            
    @property
    def name(self):
        name_elements = []
        if self.embedder is not None and self.embedder.char is not None:
            name_elements.append("Char" + self.embedder.char.arch)
        
        if self.encoders is not None:
            name_elements.append(self.encoders.arch)
            
        if self.elmo_embedder is not None:
            name_elements.append(self.elmo_embedder.arch)
            
        if self.bert_like_embedder is not None:
            name_elements.append(self.bert_like_embedder.arch)
            
        if self.intermediate is not None:
            name_elements.append(self.intermediate.arch)
            
        name_elements.append(self.decoder.arch)
        name_elements.append(self.decoder.cascade_mode)
        return '-'.join(name_elements)
    
    
    def instantiate(self, pretrained_vectors: Vectors=None, elmo: Elmo=None, bert_like: PreTrainedModel=None):
        # Only assert at the most outside level
        assert self.is_valid
        return Tagger(self, pretrained_vectors=pretrained_vectors, elmo=elmo, bert_like=bert_like)
    
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
    
    
class Tagger(nn.Module):
    def __init__(self, config: TaggerConfig, 
                 pretrained_vectors: Vectors=None, elmo: Elmo=None, bert_like: PreTrainedModel=None):
        super().__init__()
        self.config = config
        self.embedder = config.embedder.instantiate(pretrained_vectors=pretrained_vectors)
        
        if config.encoders is not None:
            self.encoders = config.encoders.instantiate()
            
        if config.elmo_embedder is not None:
            assert elmo is not None and isinstance(elmo, Elmo)
            self.elmo_embedder = config.elmo_embedder.instantiate(elmo)
            
        if config.bert_like_embedder is not None:
            assert bert_like is not None and isinstance(bert_like, PreTrainedModel)
            self.bert_like_embedder = config.bert_like_embedder.instantiate(bert_like)
            
        if config.intermediate is not None:
            self.intermediate = config.intermediate.instantiate()
            
        self.decoder = config.decoder.instantiate()
        
        
    def get_full_hidden(self, batch: Batch):
        full_hidden = []
        
        if hasattr(self, 'embedder') and hasattr(self, 'encoders'):
            embedded = self.embedder(batch)
            for encoder in self.encoders:
                full_hidden.append(encoder(batch, embedded))
                
        if hasattr(self, 'elmo_embedder'):
            full_hidden.append(self.elmo_embedder(batch))
            
        if hasattr(self, 'bert_like_embedder'):
            full_hidden.append(self.bert_like_embedder(batch))
            
        full_hidden = torch.cat(full_hidden, dim=-1)
        
        if not hasattr(self, 'intermediate'):
            return full_hidden
        else:
            return self.intermediate(batch, full_hidden)
        
        
    def forward(self, batch: Batch, return_hidden: bool=False):
        full_hidden = self.get_full_hidden(batch)
        losses = self.decoder(batch, full_hidden)
        
        # Return `hidden` for the `decode` method, to avoid duplicated computation. 
        if return_hidden:
            return losses, full_hidden
        else:
            return losses
        
        
    def decode(self, batch: Batch, full_hidden: Tensor=None):
        if full_hidden is None:
            full_hidden = self.get_full_hidden(batch)
            
        return self.decoder.decode(batch, full_hidden)

    