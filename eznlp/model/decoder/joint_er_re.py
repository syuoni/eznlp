# -*- coding: utf-8 -*-
from typing import List
import torch

from ...wrapper import Batch
from .base import DecoderMixin, DecoderConfig, Decoder
from .sequence_tagging import SequenceTaggingDecoderConfig
from .span_classification import SpanClassificationDecoderConfig
from .pair_classification import PairClassificationDecoderConfig


class JointERREDecoderMixin(DecoderMixin):
    @property
    def num_metrics(self):
        return 2
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return {**self.ck_decoder.exemplify(data_entry, training=training), 
                **self.rel_decoder.exemplify(data_entry, training=training, building=False)}
        
    def batchify(self, batch_examples: List[dict]):
        return {**self.ck_decoder.batchify(batch_examples),
                **self.rel_decoder.batchify(batch_examples)}
        
    def retrieve(self, batch: Batch):
        return self.ck_decoder.retrieve(batch), self.rel_decoder.retrieve(batch)
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        (set_chunks_gold, set_relations_gold), (set_chunks_pred, set_relations_pred) = y_gold, y_pred
        return self.ck_decoder.evaluate(set_chunks_gold, set_chunks_pred), self.rel_decoder.evaluate(set_relations_gold, set_relations_pred)



class JointERREDecoderConfig(DecoderConfig, JointERREDecoderMixin):
    def __init__(self, **kwargs):
        # It seems that pytorch does not recommend to share weights outside two modules. 
        # See https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        self.share_embeddings = kwargs.pop('share_embeddings', False)
        self.ck_decoder = kwargs.pop('ck_decoder', SpanClassificationDecoderConfig())
        self.rel_decoder = kwargs.pop('rel_decoder', PairClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def valid(self):
        return (self.ck_decoder is not None) and self.ck_decoder.valid and (self.rel_decoder is not None) and self.rel_decoder.valid
        
    @property
    def name(self):
        return self._name_sep.join([self.ck_decoder.name, self.rel_decoder.name])
        
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
        
    @property
    def in_dim(self):
        return self.ck_decoder.in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        self.ck_decoder.in_dim = dim
        self.rel_decoder.in_dim = dim
        
    def build_vocab(self, *partitions):
        self.ck_decoder.build_vocab(*partitions)
        self.rel_decoder.build_vocab(*partitions)
        
    def instantiate(self):
        return JointERREDecoder(self)



class JointERREDecoder(Decoder, JointERREDecoderMixin):
    def __init__(self, config: JointERREDecoderConfig):
        super().__init__()
        self.ck_decoder = config.ck_decoder.instantiate()
        self.rel_decoder = config.rel_decoder.instantiate()
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        ck_losses = self.ck_decoder(batch, full_hidden)
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        
        self.rel_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
        rel_losses = self.rel_decoder(batch, full_hidden)
        return ck_losses + rel_losses
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        
        self.rel_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
        batch_relations_pred = self.rel_decoder.decode(batch, full_hidden)
        return batch_chunks_pred, batch_relations_pred
