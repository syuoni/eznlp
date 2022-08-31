# -*- coding: utf-8 -*-
from typing import List, Dict
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import CombinedDropout
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from ..encoder import EncoderConfig
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .boundaries import DiagBoundariesPairs, MAX_SIZE_ID_COV_RATE, _span_pairs_from_diagonals

logger = logging.getLogger(__name__)



class DiagBoundariesPairsDecoderMixin(DecoderMixinBase):
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
        return {'dbp_obj': DiagBoundariesPairs(entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'dbp_objs': [ex['dbp_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [dbp_obj.relations for dbp_obj in batch.dbp_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']



class SpecificSpanRelClsDecoderConfig(SingleDecoderConfigBase, DiagBoundariesPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.use_biaffine = kwargs.pop('use_biaffine', True)
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 20)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 0.995)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.max_len = kwargs.pop('max_len', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
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
        return self.affine.in_dim - self.size_emb_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        if dim is not None:
            self.affine.in_dim = dim + self.size_emb_dim
        
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
        
        self.max_len = max(len(data_entry['tokens']) for data in partitions for data_entry in data)
        
        counter = Counter(label for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        counter = Counter((label, head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_self_relation = any(head[1:]==tail[1:] for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        
    def instantiate(self):
        return SpecificSpanRelClsDecoder(self)



class SpecificSpanRelClsDecoder(DecoderBase, DiagBoundariesPairsDecoderMixin):
    def __init__(self, config: SpecificSpanRelClsDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
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
        
        # TODO: Representations of context between head/tail entities
        # TODO: Chunk type embedding, only for positive (entity) spans? 
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.register_buffer('_span_size_ids', torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1))
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids > config.max_size_id, config.max_size_id)
        
        self.dropout = CombinedDropout(*config.hid_drop_rates)
        
        self.U = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim))
        self.W = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim*2))
        self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
        torch.nn.init.orthogonal_(self.U.data)
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for dbp_obj, chunks_pred in zip(batch.dbp_objs, batch_chunks_pred):
            if dbp_obj.chunks_pred is None:
                dbp_obj.chunks_pred = chunks_pred
                dbp_obj.build(self)
                dbp_obj.to(self.W.device)
        
        
    def _get_diagonal_span_size_ids(self, seq_len: int):
        span_size_ids = self._span_size_ids[:seq_len, :seq_len]
        return torch.cat([span_size_ids.diagonal(offset=k-1) for k in range(1, min(self.max_span_size, seq_len)+1)], dim=-1)
        
        
    def compute_scores(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        # full_hidden: (batch, step, hid_dim)
        # query_hidden: (batch, step-k+1, hid_dim)
        all_hidden = [full_hidden] + list(all_query_hidden.values())
        
        batch_scores = []
        for i, curr_len in enumerate(batch.seq_lens.cpu().tolist()):
            # (curr_len-k+1, hid_dim) -> (num_spans = \sum_k curr_len-k+1, hid_dim)
            span_hidden = torch.cat([all_hidden[k-1][i, :curr_len-k+1] for k in range(1, min(self.max_span_size, curr_len)+1)], dim=0)
            
            if hasattr(self, 'size_embedding'):
                # size_embedded: (num_spans, emb_dim)
                size_embedded = self.size_embedding(self._get_diagonal_span_size_ids(curr_len))
                span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
            
            if hasattr(self, 'affine_head'):
                # No mask input needed here
                affined_head = self.affine_head(span_hidden)
                affined_tail = self.affine_tail(span_hidden)
            else:
                affined_head = self.affine(span_hidden)
                affined_tail = self.affine(span_hidden)
            
            # scores1: (head_spans, affine_dim) * (voc_dim, affine_dim, affine_dim) * (affine_dim, tail_spans) -> (voc_dim, head_spans, tail_spans)
            scores1 = self.dropout(affined_head).matmul(self.U).matmul(self.dropout(affined_tail.permute(1, 0)))
            
            # affined_cat: (head_spans, tail_spans, affine_dim*2)
            affined_cat = torch.cat([self.dropout(affined_head).unsqueeze(1).expand(-1, affined_tail.size(0), -1), 
                                     self.dropout(affined_tail).unsqueeze(0).expand(affined_head.size(0), -1, -1)], dim=-1)
            
            # scores2: (voc_dim, affine_dim*2) * (head_spans, tail_spans, affine_dim*2, 1) -> (head_spans, tail_spans, voc_dim, 1) 
            scores2 = self.W.matmul(affined_cat.unsqueeze(-1))
            # scores: (head_spans, tail_spans, voc_dim)
            scores = scores1.permute(1, 2, 0) + scores2.squeeze(-1) + self.b
            batch_scores.append(scores)
        
        return batch_scores
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_scores = self.compute_scores(batch, full_hidden, all_query_hidden)
        
        losses = []
        for scores, dbp_obj in zip(batch_scores, batch.dbp_objs):
            # label_ids: (num_spans, num_spans) or (num_spans, num_spans, voc_dim)
            label_ids = dbp_obj.dbp2label_id
            if hasattr(dbp_obj, 'non_mask'):
                non_mask = dbp_obj.non_mask
                scores, label_ids = scores[non_mask], label_ids[non_mask]
            else:
                scores, label_ids = scores.flatten(end_dim=1), label_ids.flatten(end_dim=1)
            
            loss = self.criterion(scores, label_ids)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_scores = self.compute_scores(batch, full_hidden, all_query_hidden)
        
        batch_relations = []
        for scores, dbp_obj, curr_len in zip(batch_scores, batch.dbp_objs, batch.seq_lens.cpu().tolist()):
            confidences, label_ids = scores.softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
            
            relations = []
            for label, ((h_start, h_end), (t_start, t_end)) in zip(labels, _span_pairs_from_diagonals(curr_len, self.max_span_size)):
                head = (dbp_obj.span2ck_label.get((h_start, h_end), self.ck_none_label), h_start, h_end)
                tail = (dbp_obj.span2ck_label.get((t_start, t_end), self.ck_none_label), t_start, t_end)
                if label != self.none_label and head[0] != self.ck_none_label and tail[0] != self.ck_none_label:
                    relations.append((label, head, tail))
            relations = [(label, head, tail) for label, head, tail in relations 
                             if  (not self.filter_by_labels or ((label, head[0], tail[0]) in self.existing_rht_labels)) 
                             and (not self.filter_self_relation or self.existing_self_relation or (head[1:] != tail[1:]))]
            batch_relations.append(relations)
        return batch_relations
