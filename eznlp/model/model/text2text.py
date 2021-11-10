# -*- coding: utf-8 -*-
from typing import List

from ...wrapper import Batch
from ..embedder import OneHotConfig
from ..encoder import EncoderConfig
from ..decoder import GeneratorConfig
from .base import ModelConfigBase, ModelBase


class Text2TextConfig(ModelConfigBase):
    """Configurations of an text2text model. 
    
    text2text
      ├─embedder
      ├─encoder
      └─decoder(generator)
          └─embedder
    """
    
    _all_names = ['embedder', 'encoder', 'decoder']
    
    def __init__(self, **kwargs):
        self.embedder = kwargs.pop('embedder', OneHotConfig(tokens_key='tokens', field='text'))
        self.encoder = kwargs.pop('encoder', EncoderConfig(arch='LSTM'))
        self.decoder = kwargs.pop('decoder', GeneratorConfig(embedding=OneHotConfig(tokens_key='trg_tokens', field='text', has_sos=True, has_eos=True)))
        super().__init__(**kwargs)
        
    def build_vocabs_and_dims(self, *partitions):
        self.embedder.build_vocab(*partitions)
        self.decoder.build_vocab(*partitions)
        self.encoder.in_dim = self.embedder.out_dim
        self.decoder.in_dim = self.encoder.out_dim
        
    def exemplify(self, entry: dict, training: bool=True):
        example = {}
        example['tok_ids'] = self.embedder.exemplify(entry['tokens'])
        example.update(self.decoder.exemplify(entry, training=training))
        return example
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch['tok_ids'] = self.embedder.batchify([ex['tok_ids'] for ex in batch_examples])
        batch.update(self.decoder.batchify(batch_examples))
        return batch
        
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return Text2Text(self)



class Text2Text(ModelBase):
    def __init__(self, config: Text2TextConfig):
        super().__init__(config)
        
    def pretrained_parameters(self):
        params = []
        return params
        
        
    def forward2states(self, batch: Batch):
        src_embedded = self.embedder(batch.tok_ids)
        
        # src_hidden: (batch, src_step, ctx_dim)
        src_hidden = self.encoder(src_embedded, batch.mask)
        src_mask = batch.mask
        
        logits = self.decoder.forward2logits(batch, src_hidden=src_hidden, src_mask=src_mask)
        return {'src_hidden': src_hidden, 
                'src_mask': src_mask, 
                'logits': logits}
        
        
    def beam_search(self, beam_size:int, batch: Batch):
        src_embedded = self.embedder(batch.tok_ids)
        
        # src_hidden: (batch, src_step, ctx_dim)
        src_hidden = self.encoder(src_embedded, batch.mask)
        src_mask = batch.mask
        
        return self.decoder.beam_search(beam_size, batch, src_hidden=src_hidden, src_mask=src_mask)
