# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn

from ..datasets_utils import Batch
from .datasets import TagHelper
from .encoders import Encoder, RNNEncoder, CNNEncoder, TransformerEncoder, BERTEncoder
from .decoders import Decoder, SoftMaxDecoder, CRFDecoder, CascadeDecoder


class Tagger(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, batch: Batch, return_hidden: bool=False):
        full_hidden = self.encoder(batch)
        losses = self.decoder(batch, full_hidden)
        
        # Return `hidden` for the `decode` method, to avoid duplicated computation. 
        if return_hidden:
            return losses, full_hidden
        else:
            return losses
        
    def decode(self, batch: Batch, full_hidden: Tensor=None):
        if full_hidden is None:
            full_hidden = self.encoder(batch)
            
        return self.decoder.decode(batch, full_hidden)
    

def build_tagger_by_config(config: dict, tag_helper: TagHelper, itos=None, pretrained_vectors=None, bert=None):
    if 'enc' in config:
        if config['enc']['arch'].lower() in ('lstm', 'gru'):
            encoder = RNNEncoder(config, itos, pretrained_vectors)
        elif config['enc']['arch'].lower() == 'cnn':
            encoder = CNNEncoder(config, itos, pretrained_vectors)
        elif config['enc']['arch'].lower() == 'transformer':
            encoder = TransformerEncoder(config, itos, pretrained_vectors)
        else:
            raise ValueError(f"Invalid enocoder architecture {config['enc']['arch']}")
    elif 'bert' in config:
        encoder = BERTEncoder(bert, config)
    else:
        raise ValueError("Encoder architecture must be specified")
        
    if config['dec']['arch'].lower() == 'softmax':
        decoder = SoftMaxDecoder(config, tag_helper)
    elif config['dec']['arch'].lower() == 'crf':
        decoder = CRFDecoder(config, tag_helper)
    elif config['dec']['arch'].lower() == 'softmax-cascade':
        decoder = CascadeDecoder(SoftMaxDecoder(config, tag_helper))
    elif config['dec']['arch'].lower() == 'crf-cascade':
        decoder = CascadeDecoder(CRFDecoder(config, tag_helper))
    else:
        raise ValueError(f"Invalid decoder architecture {config['dec']['arch']}")
        
    return Tagger(encoder, decoder)

    