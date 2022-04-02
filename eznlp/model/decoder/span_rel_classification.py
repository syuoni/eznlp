# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import itertools
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_, reinit_vector_parameter_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import ChunkPairs

logger = logging.getLogger(__name__)


class ChunkPairsDecoderMixin(DecoderMixinBase):
    """A `Mixin` for relation extraction. 
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
        return {'cp_obj': ChunkPairs(entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'cp_objs': [ex['cp_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [cp_obj.relations for cp_obj in batch.cp_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']



class SpanRelClassificationDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
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
        self.filter_by_labels = kwargs.pop('filter_by_labels', True)
        self.filter_self_relation = kwargs.pop('filter_self_relation', True)
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
        
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        self.max_size_id = math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        
        counter = Counter(label for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        counter = Counter((label, head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_self_relation = any(head[1:]==tail[1:] for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        
    def instantiate(self):
        return SpanRelClassificationDecoder(self)



class SpanRelClassificationDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: SpanRelClassificationDecoderConfig):
        super().__init__()
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.filter_by_labels = config.filter_by_labels
        self.existing_rht_labels = config.existing_rht_labels
        self.filter_self_relation = config.filter_self_relation
        self.existing_self_relation = config.existing_self_relation
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        # Trainable context vector for overlapping chunks
        self.zero_context = torch.nn.Parameter(torch.empty(config.in_dim))
        reinit_vector_parameter_(self.zero_context)
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim*3+config.size_emb_dim*2+config.label_emb_dim*2, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for cp_obj, chunks_pred in zip(batch.cp_objs, batch_chunks_pred):
            if cp_obj.chunks_pred is None:
                cp_obj.chunks_pred = chunks_pred
                cp_obj.build(self)
                cp_obj.to(self.hid2logit.weight.device)
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for i, cp_obj in enumerate(batch.cp_objs):
            num_chunks = len(cp_obj.chunks)
            if num_chunks == 0:
                # Empty row produces loss of 0, when `criterion` uses `reduction='sum'`
                logits = torch.empty(0, self.hid2logit.out_features, device=full_hidden.device)
            else:
                # span_hidden: (num_chunks, span_size, hid_dim) -> (num_chunks, hid_dim)
                span_hidden = [full_hidden[i, start:end] for label, start, end in cp_obj.chunks]
                span_mask = seq_lens2mask(torch.tensor([h.size(0) for h in span_hidden], dtype=torch.long, device=full_hidden.device))
                span_hidden = torch.nn.utils.rnn.pad_sequence(span_hidden, batch_first=True, padding_value=0.0)
                span_hidden = self.aggregating(self.dropout(span_hidden), mask=span_mask)
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cp_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(size_embedded)], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cp_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, self.dropout(label_embedded)], dim=-1)
                
                # contexts: (num_chunks^2, ctx_span_size, hid_dim) -> (num_chunks^2, hid_dim)
                contexts = []
                for (h_label, h_start, h_end), (t_label, t_start, t_end) in itertools.product(cp_obj.chunks, cp_obj.chunks):
                    if h_end < t_start:
                        contexts.append(full_hidden[i, h_end:t_start])
                    elif t_end < h_start:
                        contexts.append(full_hidden[i, t_end:h_start])
                    else:
                        contexts.append(self.zero_context.unsqueeze(0))
                ctx_mask = seq_lens2mask(torch.tensor([c.size(0) for c in contexts], dtype=torch.long, device=full_hidden.device))
                contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True, padding_value=0.0)
                contexts = self.aggregating(self.dropout(contexts), mask=ctx_mask)
                
                # hidden_cat: (num_chunks, num_chunks, hid_dim*3)
                hidden_cat = torch.cat([span_hidden.unsqueeze(1).expand(-1, num_chunks, -1), 
                                        span_hidden.unsqueeze(0).expand(num_chunks, -1, -1), 
                                        contexts.view(num_chunks, num_chunks, -1)], dim=-1)
                # logits: (num_chunks, num_chunks, logit_dim)
                logits = self.hid2logit(hidden_cat)
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        losses = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if len(cp_obj.chunks) == 0:
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cp_obj.cp2label_id
                if hasattr(cp_obj, 'non_mask'):
                    non_mask = cp_obj.non_mask
                    logits, label_ids = logits[non_mask], label_ids[non_mask]
                else:
                    logits, label_ids = logits.flatten(end_dim=1), label_ids.flatten(end_dim=1)
                loss = self.criterion(logits, label_ids)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_relations = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if len(cp_obj.chunks) == 0:
                relations = []
            else:
                confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
                relations = [(label, head, tail) for label, (head, tail) in zip(labels, itertools.product(cp_obj.chunks, cp_obj.chunks)) if label != self.none_label]
                relations = [(label, head, tail) for label, head, tail in relations 
                                 if  (not self.filter_by_labels or ((label, head[0], tail[0]) in self.existing_rht_labels)) 
                                 and (not self.filter_self_relation or self.existing_self_relation or (head[1:] != tail[1:]))]
            batch_relations.append(relations)
        return batch_relations
