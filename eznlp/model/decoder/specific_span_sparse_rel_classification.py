# -*- coding: utf-8 -*-
from typing import List, Dict
from collections import Counter
import itertools
import logging
import copy
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import CombinedDropout
from ...nn.init import reinit_embedding_, reinit_layer_, reinit_vector_parameter_
from ..encoder import EncoderConfig
from .base import SingleDecoderConfigBase, DecoderBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import ChunkPairs
from .span_rel_classification import ChunkPairsDecoderMixin

logger = logging.getLogger(__name__)


class SpecificSpanSparseRelClsDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.use_biaffine = kwargs.pop('use_biaffine', True)
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        self.use_context = kwargs.pop('use_context', True)
        
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 20)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 0.995)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        self.label_emb_dim = kwargs.pop('label_emb_dim', 25)
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        # self.neg_sampling_power_decay = kwargs.pop('neg_sampling_power_decay', 0.0)  # decay = 0.5, 1.0
        # self.neg_sampling_surr_rate = kwargs.pop('neg_sampling_surr_rate', 0.0)
        # self.neg_sampling_surr_size = kwargs.pop('neg_sampling_surr_size', 5)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        self.filter_by_labels = kwargs.pop('filter_by_labels', True)
        self.filter_self_relation = kwargs.pop('filter_self_relation', True)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self.criterion
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'hid_drop_rates', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def in_dim(self):
        return self.affine.in_dim - self.size_emb_dim - self.label_emb_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        if dim is not None:
            self.affine.in_dim = dim + self.size_emb_dim + self.label_emb_dim
        
    @property
    def affine_ctx(self):
        affine_ctx = copy.deepcopy(self.affine)
        affine_ctx.in_dim = self.in_dim
        return affine_ctx
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())
        
        # Calculate `max_span_size` according to data
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        # Allow directly setting `max_span_size`
        if self.max_span_size is None:
            if self.max_span_size_cov_rate >= 1:
                span_size_cov = max(span_sizes)
            else:
                span_size_cov = math.ceil(numpy.quantile(span_sizes, self.max_span_size_cov_rate))
            self.max_span_size = min(span_size_cov, self.max_span_size_ceiling)
        self.max_size_id = min(math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)), self.max_span_size) - 1
        logger.warning(f"The `max_span_size` is set to {self.max_span_size}")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
        
        counter = Counter(label for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        counter = Counter((label, head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_self_relation = any(head[1:]==tail[1:] for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        
    def instantiate(self):
        return SpecificSpanSparseRelClsDecoder(self)



class SpecificSpanSparseRelClsDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: SpecificSpanSparseRelClsDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
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
        
        if config.use_biaffine:
            self.affine_head = config.affine.instantiate()
            self.affine_tail = config.affine.instantiate()
        else:
            self.affine = config.affine.instantiate()
        
        if config.use_context:
            self.affine_ctx = config.affine_ctx.instantiate()
            # Trainable context vector for overlapping chunks
            self.zero_context = torch.nn.Parameter(torch.empty(config.in_dim))
            reinit_vector_parameter_(self.zero_context)
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.dropout = CombinedDropout(*config.hid_drop_rates)
        
        self.U = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim))
        self.W = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim*(3 if config.use_context else 2)))
        self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
        torch.nn.init.orthogonal_(self.U.data)
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for cp_obj, chunks_pred in zip(batch.cp_objs, batch_chunks_pred):
            if cp_obj.chunks_pred is None:
                cp_obj.chunks_pred = chunks_pred
                cp_obj.build(self)
                cp_obj.to(self.W.device)
        
        
    def compute_scores(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        # full_hidden: (batch, step, hid_dim)
        # query_hidden: (batch, step-k+1, hid_dim)
        all_hidden = [full_hidden] + list(all_query_hidden.values())
        
        batch_scores = []
        for i, cp_obj in enumerate(batch.cp_objs):
            num_chunks = len(cp_obj.chunks)
            if num_chunks == 0:
                scores = torch.empty(0, 0, self.W.size(0), device=full_hidden.device)
            else:
                # (num_chunks, hid_dim)
                span_hidden = torch.stack([all_hidden[end-start-1][i, start] for label, start, end in cp_obj.chunks])
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cp_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cp_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, label_embedded], dim=-1)
                
                if hasattr(self, 'affine_head'):
                    # No mask input needed here
                    affined_head = self.affine_head(span_hidden)
                    affined_tail = self.affine_tail(span_hidden)
                else:
                    affined_head = self.affine(span_hidden)
                    affined_tail = self.affine(span_hidden)
                
                # scores1: (head_chunks, affine_dim) * (voc_dim, affine_dim, affine_dim) * (affine_dim, tail_chunks) -> (voc_dim, head_chunks, tail_chunks)
                scores1 = self.dropout(affined_head).matmul(self.U).matmul(self.dropout(affined_tail.permute(1, 0)))
                
                # affined_cat: (head_chunks, tail_chunks, affine_dim*2)
                affined_cat = torch.cat([self.dropout(affined_head).unsqueeze(1).expand(-1, affined_tail.size(0), -1), 
                                         self.dropout(affined_tail).unsqueeze(0).expand(affined_head.size(0), -1, -1)], dim=-1)
                
                if hasattr(self, 'affine_ctx'):
                    # contexts: (num_chunks^2, hid_dim)
                    contexts = []
                    for (h_label, h_start, h_end), (t_label, t_start, t_end) in itertools.product(cp_obj.chunks, cp_obj.chunks):
                        if h_end < t_start:
                            contexts.append(_collect_context_from_specific_span_hidden(i, h_end, t_start, all_hidden))
                        elif t_end < h_start:
                            contexts.append(_collect_context_from_specific_span_hidden(i, t_end, h_start, all_hidden))
                        else:
                            contexts.append(self.zero_context)
                    contexts = torch.stack(contexts)
                    # affined_ctx: (num_chunks^2, affine_dim) -> (num_chunks, num_chunks, affine_dim)
                    affined_ctx = self.affine_ctx(contexts).view(num_chunks, num_chunks, -1)
                    affined_cat = torch.cat([affined_cat, self.dropout(affined_ctx)], dim=-1)
                
                # scores2: (voc_dim, affine_dim*2) * (head_chunks, tail_chunks, affine_dim*2, 1) -> (head_chunks, tail_chunks, voc_dim, 1) 
                scores2 = self.W.matmul(affined_cat.unsqueeze(-1))
                # scores: (head_chunks, tail_chunks, voc_dim)
                scores = scores1.permute(1, 2, 0) + scores2.squeeze(-1) + self.b
            batch_scores.append(scores)
        
        return batch_scores
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_scores = self.compute_scores(batch, full_hidden, all_query_hidden)
        
        losses = []
        for scores, cp_obj in zip(batch_scores, batch.cp_objs):
            if len(cp_obj.chunks) == 0:
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cp_obj.cp2label_id
                if hasattr(cp_obj, 'non_mask'):
                    non_mask = cp_obj.non_mask
                    scores, label_ids = scores[non_mask], label_ids[non_mask]
                else:
                    scores, label_ids = scores.flatten(end_dim=1), label_ids.flatten(end_dim=1)
                loss = self.criterion(scores, label_ids)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_scores = self.compute_scores(batch, full_hidden, all_query_hidden)
        
        batch_relations = []
        for scores, cp_obj in zip(batch_scores, batch.cp_objs):
            if len(cp_obj.chunks) == 0:
                relations = []
            else:
                confidences, label_ids = scores.softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
                relations = [(label, head, tail) for label, (head, tail) in zip(labels, itertools.product(cp_obj.chunks, cp_obj.chunks)) if label != self.none_label]
                relations = [(label, head, tail) for label, head, tail in relations 
                                if  (not self.filter_by_labels or ((label, head[0], tail[0]) in self.existing_rht_labels)) 
                                and (not self.filter_self_relation or self.existing_self_relation or (head[1:] != tail[1:]))]
            batch_relations.append(relations)
        return batch_relations



# TODO: Aggregation?
def _collect_context_from_specific_span_hidden(i: int, start: int, end: int, all_hidden: List[torch.Tensor]):
    max_span_size = len(all_hidden)
    if end - start <= max_span_size:
        return all_hidden[end-start-1][i, start]
    else:
        return (all_hidden[max_span_size-1][i, start] + all_hidden[max_span_size-1][i, end-max_span_size]) / 2
