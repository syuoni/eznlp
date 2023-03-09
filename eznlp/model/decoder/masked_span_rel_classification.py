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


class MaskedSpanRelClsDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.fusing = kwargs.pop('fusing', 'concat')
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        self.use_context = kwargs.pop('use_context', True)
        if self.use_context:
            self.affine_ctx = copy.deepcopy(self.affine)
        
        self.size_emb_dim = kwargs.pop('size_emb_dim', 0)
        self.label_emb_dim = kwargs.pop('label_emb_dim', 0)
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
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
            if self.use_context:
                self.affine_ctx.in_dim = dim
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())
        
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        self.max_size_id = math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        
        counter = Counter(label for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        counter = Counter((label, head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_self_rel = any(head[1:]==tail[1:] for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        
    def instantiate(self):
        return MaskedSpanRelClsDecoder(self)



class MaskedSpanRelClsDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: MaskedSpanRelClsDecoderConfig):
        super().__init__()
        
        self.neg_sampling_rate = config.neg_sampling_rate
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.existing_rht_labels = config.existing_rht_labels
        self.existing_self_rel = config.existing_self_rel
        
        self.affine_head = config.affine.instantiate()
        self.affine_tail = config.affine.instantiate()
        
        if config.use_context:
            self.affine_hctx = config.affine_ctx.instantiate()
            self.affine_tctx = config.affine_ctx.instantiate()
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.dropout = CombinedDropout(*config.hid_drop_rates)
        
        if config.fusing.startswith('concat'): 
            self.hid2logit = torch.nn.Linear(config.affine.out_dim*(4 if config.use_context else 2), config.voc_dim)
            reinit_layer_(self.hid2logit, 'sigmoid')
            
        else: 
            self.Uh = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim, config.affine.out_dim))
            self.Ut = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim, config.affine.out_dim))
            self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
            torch.nn.init.orthogonal_(self.Uh.data)
            torch.nn.init.orthogonal_(self.Ut.data)
            torch.nn.init.zeros_(self.b.data)
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def compute_scores(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        # span_query_hidden/ctx_query_hidden: (batch, num_chunks, hid_dim)
        batch_scores = []
        for i, cp_obj in enumerate(batch.cp_objs):
            num_chunks = len(cp_obj.chunks)
            if num_chunks == 0:
                scores = None
            else:
                # span_hidden/ctx_hidden: (num_chunks, hid_dim)
                span_hidden = span_query_hidden[i, :num_chunks]
                ctx_hidden = ctx_query_hidden[i, :num_chunks]
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cp_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cp_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, label_embedded], dim=-1)
                
                affined_head = self.affine_head(span_hidden)
                affined_tail = self.affine_tail(span_hidden)
                
                # affined_cat: (head_chunks, tail_chunks, affine_dim*2)
                affined_cat = torch.cat([self.dropout(affined_head).unsqueeze(1).expand(-1, affined_tail.size(0), -1), 
                                         self.dropout(affined_tail).unsqueeze(0).expand(affined_head.size(0), -1, -1)], dim=-1)
                
                if hasattr(self, 'affine_hctx'):
                    affined_hctx = self.affine_hctx(ctx_hidden)
                    affined_tctx = self.affine_tctx(ctx_hidden)
                    affined_cat = torch.cat([affined_cat, 
                                             self.dropout(affined_hctx).unsqueeze(1).expand(-1, affined_tail.size(0), -1), 
                                             self.dropout(affined_tctx).unsqueeze(0).expand(affined_head.size(0), -1, -1)], dim=-1)
                
                scores = self.hid2logit(affined_cat)
            batch_scores.append(scores)
        
        return batch_scores
        
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden, span_query_hidden, ctx_query_hidden)
        
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
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden, span_query_hidden, ctx_query_hidden)
        
        batch_relations = []
        for scores, cp_obj in zip(batch_scores, batch.cp_objs):
            if len(cp_obj.chunks) == 0:
                relations = []
            else:
                confidences, label_ids = scores.softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
                relations = [(label, head, tail) for label, (head, tail) in zip(labels, itertools.product(cp_obj.chunks, cp_obj.chunks)) if label != self.none_label]
                relations = self._filter(relations)
            batch_relations.append(relations)
        
        return batch_relations
