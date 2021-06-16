# -*- coding: utf-8 -*-
from typing import List
import torch

from ...wrapper import Batch
from .base import DecoderMixin, DecoderConfig, Decoder
from .sequence_tagging import SequenceTaggingDecoderConfig
from .span_classification import SpanClassificationDecoderConfig
from .span_attr_classification import SpanAttrClassificationDecoderConfig
from .span_rel_classification import SpanRelClassificationDecoderConfig


class JointExtractionDecoderMixin(DecoderMixin):
    @property
    def has_attr_decoder(self):
        return hasattr(self, 'attr_decoder') and self.attr_decoder is not None

    @property
    def has_rel_decoder(self):
        return hasattr(self, 'rel_decoder') and self.rel_decoder is not None

    @property
    def num_metrics(self):
        return 1 + int(self.has_attr_decoder) + int(self.has_rel_decoder)

    @property
    def decoders(self):
        yield self.ck_decoder
        if self.has_attr_decoder:
            yield self.attr_decoder
        if self.has_rel_decoder:
            yield self.rel_decoder


    def exemplify(self, data_entry: dict, training: bool=True):
        example = self.ck_decoder.exemplify(data_entry, training=training)
        if self.has_attr_decoder:
            example.update(self.attr_decoder.exemplify(data_entry, training=training, building=False))
        if self.has_rel_decoder:
            example.update(self.rel_decoder.exemplify(data_entry, training=training, building=False))
        return example
        
    def batchify(self, batch_examples: List[dict]):
        batch = self.ck_decoder.batchify(batch_examples)
        if self.has_attr_decoder:
            batch.update(self.attr_decoder.batchify(batch_examples))
        if self.has_rel_decoder:
            batch.update(self.rel_decoder.batchify(batch_examples))
        return batch
        
    def retrieve(self, batch: Batch):
        return tuple(decoder.retrieve(batch) for decoder in self.decoders)
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        return tuple(decoder.evaluate(set_tuples_gold, set_tuples_pred) 
                         for decoder, set_tuples_gold, set_tuples_pred 
                         in zip(self.decoders, y_gold, y_pred))



class JointExtractionDecoderConfig(DecoderConfig, JointExtractionDecoderMixin):
    def __init__(self, **kwargs):
        # It seems that pytorch does not recommend to share weights outside two modules. 
        # See https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        self.share_embeddings = kwargs.pop('share_embeddings', False)
        self.ck_decoder = kwargs.pop('ck_decoder', SpanClassificationDecoderConfig())
        self.attr_decoder = kwargs.pop('attr_decoder', None)
        self.rel_decoder = kwargs.pop('rel_decoder', SpanRelClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def valid(self):
        return all(decoder.valid for decoder in self.decoders) and len(list(self.decoders)) >= 2
        
    @property
    def name(self):
        return self._name_sep.join([decoder.name for decoder in self.decoders])
        
    def __repr__(self):
        return self._repr_config_attrs(self.__dict__)
        
    @property
    def in_dim(self):
        return self.ck_decoder.in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        for decoder in self.decoders:
            decoder.in_dim = dim
        
    def build_vocab(self, *partitions):
        for decoder in self.decoders:
            decoder.build_vocab(*partitions)
        
    def instantiate(self):
        return JointExtractionDecoder(self)



class JointExtractionDecoder(Decoder, JointExtractionDecoderMixin):
    def __init__(self, config: JointExtractionDecoderConfig):
        super().__init__()
        self.ck_decoder = config.ck_decoder.instantiate()
        if self.has_attr_decoder:
            self.attr_decoder = config.attr_decoder.instantiate()
        if self.has_rel_decoder:
            self.rel_decoder = config.rel_decoder.instantiate()
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        losses = self.ck_decoder(batch, full_hidden)
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        
        if self.has_attr_decoder:
            self.attr_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
            losses += self.attr_decoder(batch, full_hidden)

        if self.has_rel_decoder:
            self.rel_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
            losses += self.rel_decoder(batch, full_hidden)

        return losses
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_chunks_pred = self.ck_decoder.decode(batch, full_hidden)
        y_pred = (batch_chunks_pred, )
        
        if self.has_attr_decoder:
            self.attr_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
            y_pred = (*y_pred, self.attr_decoder.decode(batch, full_hidden))

        if self.has_rel_decoder:
            self.rel_decoder.inject_chunks_and_build(batch, batch_chunks_pred, full_hidden.device)
            y_pred = (*y_pred, self.rel_decoder.decode(batch, full_hidden))
            
        return y_pred
