# -*- coding: utf-8 -*-
from typing import List, Union
import torch

from ...wrapper import Batch
from ...config import Config
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .sequence_tagging import SequenceTaggingDecoderConfig
from .span_classification import SpanClassificationDecoderConfig
from .span_attr_classification import SpanAttrClassificationDecoderConfig
from .span_rel_classification import SpanRelClassificationDecoderConfig
from .boundary_selection import BoundarySelectionDecoderConfig
from .specific_span_classification import SpecificSpanClsDecoderConfig
from .specific_span_rel_classification import SpecificSpanRelClsDecoderConfig
from .specific_span_rel_classification_unfiltered import UnfilteredSpecificSpanRelClsDecoderConfig


class JointExtractionDecoderMixin(DecoderMixinBase):
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
            example.update(self.attr_decoder.exemplify(data_entry, training=training))
        if self.has_rel_decoder:
            example.update(self.rel_decoder.exemplify(data_entry, training=training))
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



class JointExtractionDecoderConfig(Config, JointExtractionDecoderMixin):
    def __init__(self, 
                 ck_decoder: Union[SingleDecoderConfigBase, str]='span_classification', 
                 attr_decoder: Union[SingleDecoderConfigBase, str]=None, 
                 rel_decoder: Union[SingleDecoderConfigBase, str]='span_rel_classification',
                 **kwargs):
        if isinstance(ck_decoder, SingleDecoderConfigBase):
            self.ck_decoder = ck_decoder
        elif ck_decoder.lower().startswith('sequence_tagging'):
            self.ck_decoder = SequenceTaggingDecoderConfig()
        elif ck_decoder.lower().startswith('span_classification'):
            self.ck_decoder = SpanClassificationDecoderConfig()
        elif ck_decoder.lower().startswith('boundary'):
            self.ck_decoder = BoundarySelectionDecoderConfig()
        elif ck_decoder.lower().startswith('specific_span_cls'):
            self.ck_decoder = SpecificSpanClsDecoderConfig()
        
        if isinstance(attr_decoder, SingleDecoderConfigBase) or attr_decoder is None:
            self.attr_decoder = attr_decoder
        elif attr_decoder.lower().startswith('span_attr'):
            self.attr_decoder = SpanAttrClassificationDecoderConfig()
        
        if isinstance(rel_decoder, SingleDecoderConfigBase) or rel_decoder is None:
            self.rel_decoder = rel_decoder
        elif rel_decoder.lower().startswith('span_rel'):
            self.rel_decoder = SpanRelClassificationDecoderConfig()
        elif rel_decoder.lower().startswith('specific_span_rel'):
            self.rel_decoder = SpecificSpanRelClsDecoderConfig()
        elif rel_decoder.lower().startswith('unfiltered_specific_span_rel'):
            self.rel_decoder = UnfilteredSpecificSpanRelClsDecoderConfig()
        
        self.ck_loss_weight = kwargs.pop('ck_loss_weight', 1.0)
        self.attr_loss_weight = kwargs.pop('attr_loss_weight', 1.0)
        self.rel_loss_weight = kwargs.pop('rel_loss_weight', 1.0)
        
        # It seems that pytorch does not recommend to share weights outside two modules. 
        # See https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398/2
        # TODO: Refer to transformers/modeling_utils/PreTrainedModel/_tie_or_clone_weights
        self.share_embeddings = kwargs.pop('share_embeddings', False)
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
        
    @property
    def min_span_size(self):
        return self.ck_decoder.min_span_size
        
    @property
    def max_span_size(self):
        return self.ck_decoder.max_span_size
        
    @property
    def max_size_id(self):
        return self.ck_decoder.max_size_id
        
    def build_vocab(self, *partitions):
        for decoder in self.decoders:
            decoder.build_vocab(*partitions)
        
    def instantiate(self):
        return JointExtractionDecoder(self)



class JointExtractionDecoder(DecoderBase, JointExtractionDecoderMixin):
    def __init__(self, config: JointExtractionDecoderConfig):
        super().__init__()
        self.ck_decoder = config.ck_decoder.instantiate()
        self.ck_loss_weight = config.ck_loss_weight
        if config.has_attr_decoder:
            self.attr_decoder = config.attr_decoder.instantiate()
            self.attr_loss_weight = config.attr_loss_weight
        if config.has_rel_decoder:
            self.rel_decoder = config.rel_decoder.instantiate()
            self.rel_loss_weight = config.rel_loss_weight
        
        
    def forward(self, batch: Batch, **states):
        losses = self.ck_decoder(batch, **states) * self.ck_loss_weight
        batch_chunks_pred = self.ck_decoder.decode(batch, **states)
        
        if self.has_attr_decoder:
            self.attr_decoder.assign_chunks_pred(batch, batch_chunks_pred)
            losses += self.attr_decoder(batch, **states) * self.attr_loss_weight
        
        if self.has_rel_decoder:
            self.rel_decoder.assign_chunks_pred(batch, batch_chunks_pred)
            losses += self.rel_decoder(batch, **states) * self.rel_loss_weight
        
        return losses
        
        
    def decode(self, batch: Batch, **states):
        batch_chunks_pred = self.ck_decoder.decode(batch, **states)
        y_pred = (batch_chunks_pred, )
        
        if self.has_attr_decoder:
            self.attr_decoder.assign_chunks_pred(batch, batch_chunks_pred)
            y_pred = (*y_pred, self.attr_decoder.decode(batch, **states))
        
        if self.has_rel_decoder:
            self.rel_decoder.assign_chunks_pred(batch, batch_chunks_pred)
            y_pred = (*y_pred, self.rel_decoder.decode(batch, **states))
        
        return y_pred
