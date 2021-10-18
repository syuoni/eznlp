# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import torch

from ...wrapper import TargetWrapper, Batch
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase

logger = logging.getLogger(__name__)


class SpanAttrClassificationDecoderMixin(DecoderMixinBase):
    @property
    def idx2ck_label(self):
        return self._idx2ck_label
        
    @idx2ck_label.setter
    def idx2ck_label(self, idx2ck_label: List[str]):
        self._idx2ck_label = idx2ck_label
        self.ck_label2idx = {l: i for i, l in enumerate(idx2ck_label)} if idx2ck_label is not None else None
        
    @property
    def idx2attr_label(self):
        return self._idx2attr_label
        
    @idx2attr_label.setter
    def idx2attr_label(self, idx2attr_label: List[str]):
        self._idx2attr_label = idx2attr_label
        self.attr_label2idx = {l: i for i, l in enumerate(idx2attr_label)} if idx2attr_label is not None else None
        
    @property
    def ck_voc_dim(self):
        return len(self.ck_label2idx)
        
    @property
    def ck_none_idx(self):
        return self.ck_label2idx[self.ck_none_label]
        
    @property
    def attr_voc_dim(self):
        return len(self.attr_label2idx)
        
    @property
    def attr_none_idx(self):
        return self.attr_label2idx[self.attr_none_label]
        
    def exemplify(self, data_entry: dict, training: bool=True, building: bool=True):
        return {'chunks_obj': Chunks(data_entry, self, training=training, building=building)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'chunks_objs': [ex['chunks_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [chunks_obj.attributes for chunks_obj in batch.chunks_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']
        
    def inject_chunks_and_build(self, batch: Batch, batch_chunks_pred: List[List[tuple]], device=None):
        """This method is to be invoked outside, as the outside invoker is assumed not to know the structure of `batch`. 
        """
        for chunks_obj, chunks_pred in zip(batch.chunks_objs, batch_chunks_pred):
            if not chunks_obj.is_built:
                chunks_obj.inject_chunks(chunks_pred)
                chunks_obj.build(self)
                if device is not None:
                    chunks_obj.to(device)



class Chunks(TargetWrapper):
    """A wrapper of chunks with underlying attributes. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple], 
         'attributes': List[tuple]}
    
    Notes
    -----
    (1) If `building` is `True`, `data_entry['chunks']` is assumed to be known for both training and evaluation (e.g., pipeline modeling).
        In this case, the negative samples are generated from `data_entry['chunks']`. 
    (2) If `building` is `False`, `data_entry['chunks']` is known for training but not for evaluation (e.g., joint modeling).
        In this case, `inject_chunks` and `build` should be successively invoked, and the negative samples are generated from injected chunks. 
    """
    def __init__(self, data_entry: dict, config: SpanAttrClassificationDecoderMixin, training: bool=True, building: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry['chunks'] if training or building else []
        self.attributes = data_entry.get('attributes', None)
        
        self.is_built = False
        if building:
            self.build(config)
        
        
    def inject_chunks(self, chunks: List[tuple]):
        """Inject chunks from outside, typically from on-the-fly decoding. 
        
        Notes
        -----
        In merged chunks, there may exist two chunks with same spans but different chunk-types. 
        In this case, `ck_label_ids` is crucial as input. 
        """
        assert not self.is_built
        # Do not use ```self.chunks = list(set(self.chunks + chunks))```,
        # which may return non-deterministic order?
        self.chunks = self.chunks + [ck for ck in chunks if ck not in self.chunks]
        
        
    def build(self, config: SpanAttrClassificationDecoderMixin):
        """Generate negative samples from `self.chunks` and build up tensors. 
        """
        assert not self.is_built
        self.is_built = True
        
        if self.training:
            assert all(ck in self.chunks for attr_label, ck in self.attributes)
        
        # span_size_ids / ck_label_ids: (num_chunks, )
        # Note: size_id = size - 1
        self.span_size_ids = torch.tensor([end-start-1 for ck_label, start, end in self.chunks], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        self.ck_label_ids = torch.tensor([config.ck_label2idx[ck_label] for ck_label, start, end in self.chunks], dtype=torch.long)
        
        if self.attributes is not None:
            chunk2idx = {ck: k for k, ck in enumerate(self.chunks)}
            # Note: `torch.nn.BCEWithLogitsLoss` uses `float` tensor as target
            self.attr_label_ids = torch.zeros(len(self.chunks), config.attr_voc_dim, dtype=torch.float)
            for attr_label, chunk in self.attributes:
                # Note: `chunk` does not appear in `self.chunks` only if in evaluation (i.e., gold chunks missing). 
                # In this case, `attr_label_ids` is only for forward to loss, but not for backward. 
                if chunk in chunk2idx:
                    self.attr_label_ids[chunk2idx[chunk], config.attr_label2idx[attr_label]] = 1
            self.attr_label_ids[:, 0] = (self.attr_label_ids == 0).all(dim=1)




class SpanAttrClassificationDecoderConfig(SingleDecoderConfigBase, SpanAttrClassificationDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.max_span_size = kwargs.pop('max_span_size', 10)
        self.ck_size_emb_dim = kwargs.pop('ck_size_emb_dim', 25)
        self.ck_label_emb_dim = kwargs.pop('ck_label_emb_dim', 25)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        self.attr_none_label = kwargs.pop('attr_none_label', '<none>')
        self.idx2attr_label = kwargs.pop('idx2attr_label', None)
        
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
        ck_counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(ck_counter.keys())
        attr_counter = Counter(label for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.idx2attr_label = [self.attr_none_label] + list(attr_counter.keys())
        legal_ck_counter = Counter(chunk[0] for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.legal_chunk_types = set(list(legal_ck_counter.keys()))
        
    def instantiate(self):
        return SpanAttrClassificationDecoder(self)




class SpanAttrClassificationDecoder(DecoderBase, SpanAttrClassificationDecoderMixin):
    def __init__(self, config: SpanAttrClassificationDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.attr_none_label = config.attr_none_label
        self.idx2attr_label = config.idx2attr_label
        self.legal_chunk_types = config.legal_chunk_types
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        
        if config.ck_size_emb_dim > 0:
            self.ck_size_embedding = torch.nn.Embedding(config.max_span_size, config.ck_size_emb_dim)
            reinit_embedding_(self.ck_size_embedding)
        if config.ck_label_emb_dim > 0:
            self.ck_label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.ck_label_emb_dim)
            reinit_embedding_(self.ck_label_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim+config.ck_size_emb_dim+config.ck_label_emb_dim, config.attr_voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        self.confidence_threshold = config.confidence_threshold
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for k in range(full_hidden.size(0)):
            if len(batch.chunks_objs[k].chunks) == 0:
                logits = torch.empty(0, self.hid2logit.out_features, device=full_hidden.device)
                
            else:
                # span_hidden: (num_spans, span_size, hid_dim) -> (num_spans, hid_dim)
                span_hidden = [full_hidden[k, start:end] for ck_label, start, end in batch.chunks_objs[k].chunks]
                span_mask = seq_lens2mask(torch.tensor([h.size(0) for h in span_hidden], dtype=torch.long, device=full_hidden.device))
                span_hidden = torch.nn.utils.rnn.pad_sequence(span_hidden, batch_first=True, padding_value=0.0)
                span_hidden = self.aggregating(self.dropout(span_hidden), mask=span_mask)
                
                if hasattr(self, 'ck_size_embedding'):
                    # ck_size_embedded: (num_spans, emb_dim)
                    ck_size_embedded = self.ck_size_embedding(batch.chunks_objs[k].span_size_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(ck_size_embedded)], dim=-1)
                
                if hasattr(self, 'ck_label_embedding'):
                    # ck_label_embedded: (num_spans, emb_dim)
                    ck_label_embedded = self.ck_label_embedding(batch.chunks_objs[k].ck_label_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(ck_label_embedded)], dim=-1)
                
                logits = self.hid2logit(span_hidden)
            
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        losses = [self.criterion(batch_logits[k], batch.chunks_objs[k].attr_label_ids) 
                      if len(batch.chunks_objs[k].chunks) > 0
                      else torch.tensor(0.0, device=full_hidden.device)
                      for k in range(full_hidden.size(0))]
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_attributes = []
        for k in range(full_hidden.size(0)):
            attributes = []
            if len(batch.chunks_objs[k].chunks) > 0:
                for chunk, ck_sigmoids in zip(batch.chunks_objs[k].chunks, batch_logits[k].sigmoid()):
                    attr_labels = [self.idx2attr_label[i] for i, s in enumerate(ck_sigmoids.cpu().tolist()) if s >= self.confidence_threshold]
                    if self.attr_none_label not in attr_labels:
                        attributes.extend([(attr_label, chunk) for attr_label in attr_labels])
            batch_attributes.append(attributes)
        
        return batch_attributes
