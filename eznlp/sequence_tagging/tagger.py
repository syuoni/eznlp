# -*- coding: utf-8 -*-
import torch
from torchtext.experimental.vectors import Vectors
import allennlp.modules
import transformers
import flair

from ..data import Token, Batch
from ..config import Config
from ..encoder import EmbedderConfig, EncoderConfig, PreTrainedEmbedderConfig
from .decoder import DecoderConfig


class SequenceTaggerConfig(Config):
    def __init__(self, **kwargs):
        """
        Configurations of a sequence tagger. 
        
        tagger
          ├─decoder
          └─intermediate
              ├─encoder
              │   └─embedder
              ├─elmo_embedder
              ├─bert_like_embedder
              ├─flair_fw_embedder
              └─flair_bw_embedder
        """
        self.embedder: EmbedderConfig = kwargs.pop('embedder', EmbedderConfig())
        self.encoder: EncoderConfig = kwargs.pop('encoder', EncoderConfig(arch='LSTM'))
        
        self.elmo_embedder: PreTrainedEmbedderConfig = kwargs.pop('elmo_embedder', None)
        self.bert_like_embedder: PreTrainedEmbedderConfig = kwargs.pop('bert_like_embedder', None)
        self.flair_fw_embedder: PreTrainedEmbedderConfig = kwargs.pop('flair_fw_embedder', None)
        self.flair_bw_embedder: PreTrainedEmbedderConfig = kwargs.pop('flair_bw_embedder', None)
        
        self.intermediate: EncoderConfig = kwargs.pop('intermediate', None)
        self.decoder: DecoderConfig = kwargs.pop('decoder', DecoderConfig(arch='CRF'))
        super().__init__(**kwargs)
        
    @property
    def necessary_assemblies(self):
        return [self.embedder, self.decoder]
        
    @property
    def optional_assemblies(self):
        optional_assemblies = [self.encoder, self.elmo_embedder, self.bert_like_embedder, self.flair_fw_embedder, self.flair_bw_embedder]
        return [assembly for assembly in optional_assemblies if assembly is not None]
        
    @property
    def is_valid(self):
        for assembly in self.necessary_assemblies:
            if assembly is None or not assembly.is_valid:
                return False
        
        for assembly in self.optional_assemblies:
            if not assembly.is_valid:
                return False
            
        if self.intermediate is not None and not self.intermediate.is_valid:
            return False
            
        return True
    
    
    def _update_dims(self, ex_token: Token=None):
        if self.embedder.val is not None and ex_token is not None:
            for f, val_config in self.embedder.val.items():
                val_config.in_dim = getattr(ex_token, f).shape[0]
                
        if self.encoder is not None:
            self.encoder.in_dim = self.embedder.out_dim
            
        full_hid_dim = sum(assembly.out_dim for assembly in self.optional_assemblies)
        
        if self.intermediate is None:
            self.decoder.in_dim = full_hid_dim
        else:
            self.intermediate.in_dim = full_hid_dim
            self.decoder.in_dim = self.intermediate.out_dim
            
            
    @property
    def name(self):
        name_elements = []
        if self.embedder.char is not None:
            name_elements.append("Char" + self.embedder.char.arch)
        
        for assembly in self.optional_assemblies:
            name_elements.append(assembly.arch)
            
        if self.intermediate is not None:
            name_elements.append(self.intermediate.arch)
        
        name_elements.append(self.decoder.arch)
        # name_elements.append(self.decoder.cascade_mode)
        return '-'.join(name_elements)
    
    
    def instantiate(self, 
                    pretrained_vectors: Vectors=None, 
                    elmo: allennlp.modules.elmo.Elmo=None, 
                    bert_like: transformers.PreTrainedModel=None, 
                    flair_fw_lm: flair.models.LanguageModel=None, 
                    flair_bw_lm: flair.models.LanguageModel=None):
        # Only check validity at the most outside level
        assert self.is_valid
        return SequenceTagger(self, pretrained_vectors, elmo, bert_like, flair_fw_lm, flair_bw_lm)
    
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
    
    
    
class SequenceTagger(torch.nn.Module):
    def __init__(self, config: SequenceTaggerConfig, 
                 pretrained_vectors: Vectors=None, 
                 elmo: allennlp.modules.elmo.Elmo=None, 
                 bert_like: transformers.PreTrainedModel=None, 
                 flair_fw_lm: flair.models.LanguageModel=None, 
                 flair_bw_lm: flair.models.LanguageModel=None):
        super().__init__()
        self.embedder = config.embedder.instantiate(pretrained_vectors=pretrained_vectors)
        
        if config.encoder is not None:
            self.encoder = config.encoder.instantiate()
            
        if config.elmo_embedder is not None:
            assert elmo is not None and isinstance(elmo, allennlp.modules.elmo.Elmo)
            self.elmo_embedder = config.elmo_embedder.instantiate(elmo)
            
        if config.bert_like_embedder is not None:
            assert bert_like is not None and isinstance(bert_like, transformers.PreTrainedModel)
            self.bert_like_embedder = config.bert_like_embedder.instantiate(bert_like)
            
        if config.flair_fw_embedder is not None:
            assert flair_fw_lm is not None and isinstance(flair_fw_lm, flair.models.LanguageModel)
            self.flair_fw_embedder = config.flair_fw_embedder.instantiate(flair_fw_lm)
            
        if config.flair_bw_embedder is not None:
            assert flair_bw_lm is not None and isinstance(flair_bw_lm, flair.models.LanguageModel)
            self.flair_bw_embedder = config.flair_bw_embedder.instantiate(flair_bw_lm)
            
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
            
        if hasattr(self, 'flair_fw_embedder'):
            full_hidden.append(self.flair_fw_embedder(batch))
        if hasattr(self, 'flair_bw_embedder'):
            full_hidden.append(self.flair_bw_embedder(batch))
            
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

    