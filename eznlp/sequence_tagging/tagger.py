# -*- coding: utf-8 -*-
import torch
from torchtext.experimental.vectors import Vectors
import allennlp.modules
import transformers
import flair

from ..token import Token
from ..dataset_utils import Batch
from ..config import Config
from ..encoder import EmbedderConfig, EncoderConfig, PreTrainedEmbedderConfig
from .decoder import DecoderConfig


class SequenceTaggerConfig(Config):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        embedder: EmbedderConfig
        encoder: EncoderConfig
        elmo_embedder: PreTrainedEmbedderConfig
        bert_like_embedder: PreTrainedEmbedderConfig
        flair_embedder: PreTrainedEmbedderConfig
        intermediate: EncoderConfig
        decoder: DecoderConfig
        """
        self.embedder = kwargs.pop('embedder', EmbedderConfig())
        self.encoder = kwargs.pop('encoder', EncoderConfig(arch='LSTM'))
        
        self.elmo_embedder = kwargs.pop('elmo_embedder', None)
        self.bert_like_embedder = kwargs.pop('bert_like_embedder', None)
        self.flair_embedder = kwargs.pop('flair_embedder', None)
        
        self.intermediate = kwargs.pop('intermediate', None)
        self.decoder = kwargs.pop('decoder', DecoderConfig(arch='CRF'))
        super().__init__(**kwargs)
        
        
    @property
    def is_valid(self):
        for assembly in [self.embedder, self.decoder]:
            if assembly is None or not assembly.is_valid:
                return False
        
        for assembly in [self.encoder, self.elmo_embedder, self.bert_like_embedder, self.flair_embedder, self.intermediate]:
            if assembly is not None and not assembly.is_valid:
                return False
            
        return True
        
    
    def _update_dims(self, ex_token: Token=None):
        if self.embedder.val is not None and ex_token is not None:
            for f, val_config in self.embedder.val.items():
                val_config.in_dim = getattr(ex_token, f).shape[0]
                
        if self.encoder is not None:
            self.encoder.in_dim = self.embedder.out_dim
            
        full_hid_dim = 0
        for assembly in [self.encoder, self.elmo_embedder, self.bert_like_embedder, self.flair_embedder]:
            if assembly is not None:
                full_hid_dim += assembly.out_dim
        
        if self.intermediate is None:
            self.decoder.in_dim = full_hid_dim
        else:
            self.intermediate.in_dim = full_hid_dim
            self.decoder.in_dim = self.intermediate.out_dim
            
            
    @property
    def name(self):
        name_elements = []
        if self.embedder is not None and self.embedder.char is not None:
            name_elements.append("Char" + self.embedder.char.arch)
        
        for assembly in [self.encoder, self.elmo_embedder, self.bert_like_embedder, self.flair_embedder, self.intermediate]:
            if assembly is not None:
                name_elements.append(assembly.arch)
                
        name_elements.append(self.decoder.arch)
        name_elements.append(self.decoder.cascade_mode)
        return '-'.join(name_elements)
    
    
    def instantiate(self, 
                    pretrained_vectors: Vectors=None, 
                    elmo: allennlp.modules.elmo.Elmo=None, 
                    bert_like: transformers.PreTrainedModel=None, 
                    flair_emb: flair.embeddings.TokenEmbeddings=None):
        # Only check validity at the most outside level
        assert self.is_valid
        return SequenceTagger(self, 
                              pretrained_vectors=pretrained_vectors, 
                              elmo=elmo, 
                              bert_like=bert_like, 
                              flair_emb=flair_emb)
    
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
    
    
    
    
class SequenceTagger(torch.nn.Module):
    def __init__(self, 
                 config: SequenceTaggerConfig, 
                 pretrained_vectors: Vectors=None, 
                 elmo: allennlp.modules.elmo.Elmo=None, 
                 bert_like: transformers.PreTrainedModel=None, 
                 flair_emb: flair.embeddings.TokenEmbeddings=None):
        super().__init__()
        # self.config = config
        self.embedder = config.embedder.instantiate(pretrained_vectors=pretrained_vectors)
        
        if config.encoder is not None:
            self.encoder = config.encoder.instantiate()
            
        if config.elmo_embedder is not None:
            assert elmo is not None and isinstance(elmo, allennlp.modules.elmo.Elmo)
            self.elmo_embedder = config.elmo_embedder.instantiate(elmo)
            
        if config.bert_like_embedder is not None:
            assert bert_like is not None and isinstance(bert_like, transformers.PreTrainedModel)
            self.bert_like_embedder = config.bert_like_embedder.instantiate(bert_like)
            
        if config.flair_embedder is not None:
            assert flair_emb is not None and isinstance(flair_emb, flair.embeddings.TokenEmbeddings)
            self.flair_embedder = config.flair_embedder.instantiate(flair_emb)
            
        if config.intermediate is not None:
            self.intermediate = config.intermediate.instantiate()
            
        self.decoder = config.decoder.instantiate()
        
        
    def get_full_hidden(self, batch: Batch):
        full_hidden = []
        
        if hasattr(self, 'embedder') and hasattr(self, 'encoder'):
            embedded = self.embedder(batch)
            full_hidden.append(self.encoder(batch, embedded))
            
        if hasattr(self, 'elmo_embedder'):
            full_hidden.append(self.elmo_embedder(batch))
            
        if hasattr(self, 'bert_like_embedder'):
            full_hidden.append(self.bert_like_embedder(batch))
            
        if hasattr(self, 'flair_embedder'):
            full_hidden.append(self.flair_embedder(batch))
            
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
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor=None):
        if full_hidden is None:
            full_hidden = self.get_full_hidden(batch)
            
        return self.decoder.decode(batch, full_hidden)

    