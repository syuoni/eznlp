# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
from torchtext.experimental.vectors import Vectors
from transformers import PreTrainedModel

from ..datasets_utils import Batch
from .config import TaggerConfig
from ..embedders import Embedder
from ..encoders import ShortcutEncoder, RNNEncoder, CNNEncoder, TransformerEncoder, PreTrainedEncoder
from .decoders import SoftMaxDecoder, CRFDecoder, CascadeDecoder


class Tagger(nn.Module):
    def __init__(self, config: TaggerConfig, pretrained_vectors: Vectors=None, ptm: PreTrainedModel=None):
        super().__init__()
        self.config = config
        self.embedder = Embedder(config.embedder, pretrained_vectors=pretrained_vectors)
        
        # NOTE: The order should be preserved. 
        if config.encoders is not None:
            encoders = []
            for enc_config in config.encoders:
                if enc_config.arch.lower() == 'shortcut':
                    encoders.append(ShortcutEncoder(enc_config))
                elif enc_config.arch.lower() in ('lstm', 'gru'):
                    encoders.append(RNNEncoder(enc_config))
                elif enc_config.arch.lower() == 'cnn':
                    encoders.append(CNNEncoder(enc_config))
                elif enc_config.arch.lower() == 'transformer':
                    encoders.append(TransformerEncoder(enc_config))
                else:
                    raise ValueError(f"Invalid enocoder architecture {enc_config.arch}")
            self.encoders = nn.ModuleList(encoders)
        
        if config.ptm_encoder is not None:
            assert ptm is not None
            self.ptm_encoder = PreTrainedEncoder(ptm)
        
        if config.decoder.cascade_mode.lower() == 'none':
            if config.decoder.arch.lower() == 'softmax':
                self.decoder = SoftMaxDecoder(config.decoder)
            elif config.decoder.arch.lower() == 'crf':
                self.decoder = CRFDecoder(config.decoder)
            else:
                raise ValueError(f"Invalid decoder architecture {config.decoder.arch}")
        else:
            self.decoder = CascadeDecoder(config.decoder)
            
            
    def get_full_hidden(self, batch: Batch):
        full_hidden = []
        
        if hasattr(self, 'embedder') and hasattr(self, 'encoders'):
            embedded = self.embedder(batch, word=True, char=True, enum=True, val=True)
            for encoder in self.encoders:
                full_hidden.append(encoder(batch, embedded))
                
        if hasattr(self, 'ptm_encoder'):
            full_hidden.append(self.ptm_encoder(batch))
            
        return torch.cat(full_hidden, dim=-1)
            
            
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

    