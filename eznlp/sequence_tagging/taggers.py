# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn

from ..datasets_utils import Batch
from .datasets import TagHelper
from ..embedders import Embedder
from ..encoders import ShortcutEncoder, RNNEncoder, CNNEncoder, TransformerEncoder, PreTrainedEncoder
from .decoders import SoftMaxDecoder, CRFDecoder, CascadeDecoder


class Tagger(nn.Module):
    def __init__(self, config: dict, tag_helper: TagHelper, itos=None, pretrained_vectors=None, ptm=None):
        super().__init__()
        self.config = config
        
        assert ('emb' in config and 'enc' in config) or ('pt_enc' in config)
        if 'emb' in config:
            self.embedder = Embedder(config['emb'], itos, pretrained_vectors)
            
        if 'enc' in config:
            encoders = []
            for enc_config in config['enc']:
                if enc_config['arch'].lower() == 'shortcut':
                    encoder = ShortcutEncoder(enc_config)
                elif enc_config['arch'].lower() in ('lstm', 'gru'):
                    encoder = RNNEncoder(enc_config)
                elif enc_config['arch'].lower() == 'cnn':
                    encoder = CNNEncoder(enc_config)
                elif enc_config['arch'].lower() == 'transformer':
                    encoder = TransformerEncoder(enc_config)
                else:
                    raise ValueError(f"Invalid enocoder architecture {enc_config['arch']}")
                encoders.append(encoder)
            self.encoders = nn.ModuleList(encoders)
        
        if 'pt_enc' in config:
            assert ptm is not None
            self.pt_encoder = PreTrainedEncoder(ptm)
        
        
        assert 'dec' in config
        dec_config = config['dec']
        if dec_config['arch'].lower() == 'softmax':
            decoder = SoftMaxDecoder(dec_config, tag_helper)
        elif dec_config['arch'].lower() == 'crf':
            decoder = CRFDecoder(dec_config, tag_helper)
        elif dec_config['arch'].lower() == 'softmax-cascade':
            decoder = CascadeDecoder(SoftMaxDecoder(dec_config, tag_helper))
        elif dec_config['arch'].lower() == 'crf-cascade':
            decoder = CascadeDecoder(CRFDecoder(dec_config, tag_helper))
        else:
            raise ValueError(f"Invalid decoder architecture {dec_config['arch']}")
        self.decoder = decoder
        
        
    def get_full_hidden(self, batch: Batch):
        full_hidden = []
        
        if hasattr(self, 'embedder') and hasattr(self, 'encoders'):
            embedded = self.embedder(batch, word=True, char_cnn=True, enum=True, val=True)
            for encoder in self.encoders:
                full_hidden.append(encoder(batch, embedded))
                
        if hasattr(self, 'pt_encoder'):
            full_hidden.append(self.pt_encoder(batch))
            
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

    