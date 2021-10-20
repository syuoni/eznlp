# -*- coding: utf-8 -*-
from typing import List

from ...wrapper import Batch
from ..embedder import OneHotConfig
from ..image_encoder import ImageEncoderConfig
from ..decoder import GeneratorConfig
from .base import ModelConfigBase, ModelBase


class Image2TextConfig(ModelConfigBase):
    """Configurations of an image2text model. 
    
    image2text
      ├─encoder
      └─decoder(generator)
          └─embedder
    """
    
    _all_names = ['encoder', 'decoder']
    
    def __init__(self, **kwargs):
        self.encoder = kwargs.pop('encoder')
        self.decoder = kwargs.pop('decoder', GeneratorConfig(embedding=OneHotConfig(tokens_key='trg_tokens', field='text', has_sos=True, has_eos=True)))
        super().__init__(**kwargs)
        
    def build_vocabs_and_dims(self, *partitions):
        self.decoder.build_vocab(*partitions)
        self.decoder.in_dim = self.encoder.out_dim
        
    def exemplify(self, entry: dict, training: bool=True):
        example = {}
        example.update(self.encoder.exemplify(entry, training=training))
        example.update(self.decoder.exemplify(entry, training=training))
        return example
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch.update(self.encoder.batchify(batch_examples))
        batch.update(self.decoder.batchify(batch_examples))
        return batch
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return Image2Text(self)



class Image2Text(ModelBase):
    def __init__(self, config: Image2TextConfig):
        super().__init__(config)
        
    def pretrained_parameters(self):
        params = []
        params.extend(self.encoder.backbone.parameters())
        return params
        
        
    def forward2states(self, batch: Batch):
        # src_hidden: (batch, ctx_dim, height, width) -> (batch, src_step, ctx_dim)
        src_hidden = self.encoder(batch.img)
        src_hidden = src_hidden.flatten(start_dim=2).permute(0, 2, 1)
        src_mask = None
        
        logits = self.decoder.forward2logits(batch, src_hidden=src_hidden, src_mask=src_mask)
        return {'src_hidden': src_hidden, 
                'src_mask': src_mask, 
                'logits': logits}
        
        
    def beam_search(self, beam_size:int, batch: Batch):
        # src_hidden: (batch, ctx_dim, height, width) -> (batch, src_step, ctx_dim)
        src_hidden = self.encoder(batch.img)
        src_hidden = src_hidden.flatten(start_dim=2).permute(0, 2, 1)
        src_mask = None
        
        return self.decoder.beam_search(beam_size, batch, src_hidden=src_hidden, src_mask=src_mask)
