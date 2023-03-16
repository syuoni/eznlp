# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import copy
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import CombinedDropout
from ...nn.modules import BiAffineFusor, TriAffineFusor
from ...nn.init import reinit_embedding_, reinit_layer_, reinit_vector_parameter_
from ...utils.chunk import chunk_pair_distance
from ..encoder import EncoderConfig
from .base import SingleDecoderConfigBase, DecoderBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import ChunkPairs
from .span_rel_classification import ChunkPairsDecoderMixin

logger = logging.getLogger(__name__)


class MaskedSpanRelClsDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.use_context = kwargs.pop('use_context', True)
        self.context_mode = kwargs.pop('context_mode', 'specific')
        assert not (self.use_context and self.context_mode.lower().count('none'))
        
        self.size_emb_dim = kwargs.pop('size_emb_dim', 0)
        self.label_emb_dim = kwargs.pop('label_emb_dim', 0)
        self.fusing_mode = kwargs.pop('fusing_mode', 'concat')
        self.reduction = kwargs.pop('reduction', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0))
        if self.use_context: 
            self.reduction_ctx = copy.deepcopy(self.reduction)
        
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        
        self.ck_loss_weight = kwargs.pop('ck_loss_weight', 0)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.fusing_mode, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'hid_drop_rates', 'fusing_mode', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def in_dim(self): 
        return self._in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int): 
        if dim is not None: 
            self._in_dim = dim
            self.reduction.in_dim = dim + self.size_emb_dim + self.label_emb_dim
            if self.use_context:
                self.reduction_ctx.in_dim = dim
        
        
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
        self.max_cp_distance = max(chunk_pair_distance(head, tail) for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        
    def instantiate(self):
        return MaskedSpanRelClsDecoder(self)



class MaskedSpanRelClsDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: MaskedSpanRelClsDecoderConfig):
        super().__init__()
        self.use_context = config.use_context
        self.context_mode = config.context_mode
        self.fusing_mode = config.fusing_mode
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.ck_loss_weight = config.ck_loss_weight
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.existing_rht_labels = config.existing_rht_labels
        self.existing_self_rel = config.existing_self_rel
        self.max_cp_distance = config.max_cp_distance
        
        if config.use_context: 
            # Trainable context vector for overlapping chunk pairs
            self.zero_context = torch.nn.Parameter(torch.empty(config.in_dim))
            reinit_vector_parameter_(self.zero_context)
            # A placeholder context vector for invalid chunk pairs
            self.register_buffer('none_context', torch.zeros(config.in_dim))
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)
        
        if config.fusing_mode.lower().startswith('concat'):
            if config.use_context: 
                self.hid2logit = torch.nn.Linear(config.in_dim*4+config.size_emb_dim*2+config.label_emb_dim*2, config.voc_dim)
            else:
                self.hid2logit = torch.nn.Linear(config.in_dim*2+config.size_emb_dim*2+config.label_emb_dim*2, config.voc_dim)
            reinit_layer_(self.hid2logit, 'sigmoid')
            
        elif config.fusing_mode.lower().startswith('affine'):
            self.reduction_head = config.reduction.instantiate()
            self.reduction_tail = config.reduction.instantiate()
            if config.use_context: 
                self.reduction_hctx = config.reduction_ctx.instantiate()
                self.reduction_tctx = config.reduction_ctx.instantiate()
                self.hid2logit_hctx = TriAffineFusor(config.reduction.out_dim, config.voc_dim)
                self.hid2logit_tctx = TriAffineFusor(config.reduction.out_dim, config.voc_dim)
            else:
                self.hid2logit = BiAffineFusor(config.reduction.out_dim, config.voc_dim)
        
        if config.ck_loss_weight > 0: 
            self.ck_hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim+config.label_emb_dim, config.ck_voc_dim)
            reinit_layer_(self.ck_hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        # span_query_hidden/ctx_query_hidden: (batch, num_chunks, hid_dim)
        batch_logits = []
        for i, cp_obj in enumerate(batch.cp_objs):
            if not cp_obj.has_valid_cp: 
                logits = None
            else:
                # span_hidden/ctx_hidden: (num_chunks, hid_dim)
                span_hidden = span_query_hidden[i, :len(cp_obj.chunks)]
                ctx_hidden = ctx_query_hidden[i, :len(cp_obj.chunks)]
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cp_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cp_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, label_embedded], dim=-1)
                
                head_hidden = self.in_dropout(span_hidden).unsqueeze(1).expand(-1, len(cp_obj.chunks), -1)
                tail_hidden = self.in_dropout(span_hidden).unsqueeze(0).expand(len(cp_obj.chunks), -1, -1)
                hctx_hidden = self.in_dropout(ctx_hidden).unsqueeze(1).expand(-1, len(cp_obj.chunks), -1)
                tctx_hidden = self.in_dropout(ctx_hidden).unsqueeze(0).expand(len(cp_obj.chunks), -1, -1)
                
                if self.fusing_mode.startswith('concat'):
                    # hidden_cat: (num_chunks, num_chunks, hid_dim*4)
                    hidden_cat = torch.cat([head_hidden, tail_hidden], dim=-1)
                    if self.use_context: 
                        hidden_cat = torch.cat([hidden_cat, hctx_hidden, tctx_hidden], dim=-1)
                    # logits: (num_chunks, num_chunks, logit_dim)
                    logits = self.hid2logit(hidden_cat)
                    
                elif self.fusing_mode.lower().startswith('affine'):
                    reduced_head = self.reduction_head(head_hidden)
                    reduced_tail = self.reduction_tail(tail_hidden)
                    if self.use_context: 
                        reduced_hctx = self.reduction_hctx(hctx_hidden)
                        reduced_tctx = self.reduction_tctx(tctx_hidden)
                        logits_hctx = self.hid2logit_hctx(self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail), self.hid_dropout(reduced_hctx))
                        logits_tctx = self.hid2logit_tctx(self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail), self.hid_dropout(reduced_tctx))
                        logits = (logits_hctx + logits_tctx) * (0.5**0.5)
                    else:
                        logits = self.hid2logit(self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail))
                
                if self.ck_loss_weight > 0:
                    ck_logits = self.ck_hid2logit(span_hidden)
                    logits = (logits, ck_logits)
            
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden, span_query_hidden, ctx_query_hidden)
        
        losses = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if not cp_obj.has_valid_cp: 
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cp_obj.cp2label_id
                ck_label_ids_gold = cp_obj.ck_label_ids_gold
                if self.ck_loss_weight > 0: 
                    logits, ck_logits = logits
                logits, label_ids = logits[cp_obj.non_mask], label_ids[cp_obj.non_mask]
                loss = self.criterion(logits, label_ids)
                if self.ck_loss_weight > 0: 
                    loss = loss + self.ck_loss_weight*self.criterion(ck_logits, ck_label_ids_gold)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor, ctx_query_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden, span_query_hidden, ctx_query_hidden)
        
        batch_relations = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if not cp_obj.has_valid_cp: 
                relations = []
            else:
                if self.ck_loss_weight > 0:
                    logits, _ = logits
                confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
                relations = [(label, head, tail) for label, (head, tail, is_valid) in zip(labels, self.enumerate_chunk_pairs(cp_obj)) if is_valid and label != self.none_label]
                relations = self._filter(relations)
            batch_relations.append(relations)
        
        return batch_relations
