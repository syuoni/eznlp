# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .chunks import ChunkSingles

logger = logging.getLogger(__name__)


class ChunkSinglesDecoderMixin(DecoderMixinBase):
    """A `Mixin` for attribute extraction. 
    """
    @property
    def idx2label(self):
        return self._idx2label
        
    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        
    @property
    def voc_dim(self):
        return len(self.label2idx)
        
    @property
    def none_idx(self):
        return self.label2idx[self.none_label]
        
    @property
    def idx2ck_label(self):
        return self._idx2ck_label
        
    @idx2ck_label.setter
    def idx2ck_label(self, idx2ck_label: List[str]):
        self._idx2ck_label = idx2ck_label
        self.ck_label2idx = {l: i for i, l in enumerate(idx2ck_label)} if idx2ck_label is not None else None
        
    @property
    def ck_voc_dim(self):
        return len(self.ck_label2idx)
        
    @property
    def ck_none_idx(self):
        return self.ck_label2idx[self.ck_none_label]
        
    def exemplify(self, entry: dict, training: bool=True):
        return {'cs_obj': ChunkSingles(entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'cs_objs': [ex['cs_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [cs_obj.attributes for cs_obj in batch.cs_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']



class SpanAttrClassificationDecoderConfig(SingleDecoderConfigBase, ChunkSinglesDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        # Note: The spans with sizes longer than `max_span_size` will be masked/ignored in both training and inference. 
        # Hence, these spans will never be recalled in testing. 
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 50)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 1.0)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        self.label_emb_dim = kwargs.pop('label_emb_dim', 25)
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        # self.neg_sampling_power_decay = kwargs.pop('neg_sampling_power_decay', 0.0)  # decay = 0.5, 1.0
        # self.neg_sampling_surr_rate = kwargs.pop('neg_sampling_surr_rate', 0.0)
        # self.neg_sampling_surr_size = kwargs.pop('neg_sampling_surr_size', 5)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        
        self.multihot = True
        self.confidence_threshold = kwargs.pop('confidence_threshold', 0.5)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'agg_mode', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())
        
        # Allow directly setting `max_span_size`
        if self.max_span_size is None:
            # Calculate `max_span_size` according to data
            span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
            if self.max_span_size_cov_rate >= 1:
                span_size_cov = max(span_sizes)
            else:
                span_size_cov = math.ceil(numpy.quantile(span_sizes, self.max_span_size_cov_rate))
            self.max_span_size = min(span_size_cov, self.max_span_size_ceiling)
        logger.warning(f"The `max_span_size` is set to {self.max_span_size}")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
        
        counter = Counter(label for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        # TODO: allow_ck2attr_type
        counter = Counter(chunk[0] for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.allow_ck_types = set(counter.keys())
        
        
    def instantiate(self):
        return SpanAttrClassificationDecoder(self)



class SpanAttrClassificationDecoder(DecoderBase, ChunkSinglesDecoderMixin):
    def __init__(self, config: SpanAttrClassificationDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
        self.neg_sampling_rate = config.neg_sampling_rate
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.allow_ck_types = config.allow_ck_types
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_span_size, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim+config.label_emb_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        self.confidence_threshold = config.confidence_threshold
        
        
    def _assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for cs_obj, chunks_pred in zip(batch.cs_objs, batch_chunks_pred):
            cs_obj.chunks_pred = chunks_pred
            cs_obj.build(self)
            cs_obj.to(self.hid2logit.weight.device)
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for i, cs_obj in enumerate(batch.cs_objs):
            num_chunks = len(cs_obj.chunks)
            if num_chunks == 0:
                # Empty row produces loss of 0, when `criterion` uses `reduction='sum'`
                logits = torch.empty(0, self.hid2logit.out_features, device=full_hidden.device)
            else:
                # span_hidden: (num_chunks, span_size, hid_dim) -> (num_chunks, hid_dim)
                span_hidden = [full_hidden[i, start:end] for label, start, end in cs_obj.chunks]
                span_mask = seq_lens2mask(torch.tensor([h.size(0) for h in span_hidden], dtype=torch.long, device=full_hidden.device))
                span_hidden = torch.nn.utils.rnn.pad_sequence(span_hidden, batch_first=True, padding_value=0.0)
                span_hidden = self.aggregating(self.dropout(span_hidden), mask=span_mask)
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cs_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(size_embedded)], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cs_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(label_embedded)], dim=-1)
                
                # logits: (num_chunks, logit_dim)
                logits = self.hid2logit(span_hidden)
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        losses = []
        for logits, cs_obj in zip(batch_logits, batch.cs_objs):
            label_ids = cs_obj.cs2label_id
            if hasattr(cs_obj, 'non_mask'):
                non_mask = cs_obj.non_mask
                logits, label_ids = logits[non_mask], label_ids[non_mask]
            
            loss = self.criterion(logits, label_ids) if len(cs_obj.chunks) > 0 else torch.tensor(0.0, device=full_hidden.device)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_attributes = []
        for logits, cs_obj in zip(batch_logits, batch.cs_objs):
            confidences = logits.sigmoid()
            
            attributes = []
            for chunk, ck_confidences in zip(cs_obj.chunks, confidences):
                labels = [self.idx2label[i] for i, c in enumerate(ck_confidences.cpu().tolist()) if c >= self.confidence_threshold]
                if self.none_label not in labels:
                    attributes.extend([(label, chunk) for label in labels])
            batch_attributes.append(attributes)
        return batch_attributes
