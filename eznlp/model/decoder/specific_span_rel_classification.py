# -*- coding: utf-8 -*-
from typing import List, Dict
from collections import Counter
import logging
import copy
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.modules import SoftLabelCrossEntropyLoss, BiAffineFusor, TriAffineFusor
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_, reinit_vector_parameter_
from ...utils.chunk import chunk_pair_distance
from ...utils.relation import INV_REL_PREFIX
from ..encoder import EncoderConfig
from .base import SingleDecoderConfigBase, DecoderBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import ChunkPairs, MAX_DIST_ID_COV_RATE
from .span_rel_classification import ChunkPairsDecoderMixin

logger = logging.getLogger(__name__)


class SpecificSpanRelClsDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
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
        if self.fusing_mode.lower().startswith('concat'):
            self.reduction_cat = copy.deepcopy(self.reduction)
            self.reduction_cat.hid_dim = self.reduction.hid_dim*self.num_hidden
        
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.min_span_size = kwargs.pop('min_span_size', 2)
        assert self.min_span_size in (1, 2)
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 20)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 0.995)
        self.max_span_size = kwargs.pop('max_span_size', None)
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        self.ck_loss_weight = kwargs.pop('ck_loss_weight', 0)
        self.l2_loss_weight = kwargs.pop('l2_loss_weight', 0)
        
        self.sym_rel_labels = kwargs.pop('sym_rel_labels', [])
        self.comp_sym_rel = kwargs.pop('comp_sym_rel', False)
        self.use_inv_rel = kwargs.pop('use_inv_rel', False)
        self.check_rht_labels = kwargs.pop('check_rht_labels', False)
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        
        # Simplified smoothing epsilon
        self.ss_epsilon = kwargs.pop('ss_epsilon', 0.0)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.fusing_mode, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'hid_drop_rates', 'agg_mode', 'fusing_mode', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def num_hidden(self):
        return 3 if self.use_context else 2
        
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
            if self.fusing_mode.lower().startswith('concat'):
                self.reduction_cat.in_dim = self.in_dim*self.num_hidden + self.size_emb_dim*2 + self.label_emb_dim*2
        
    @property
    def criterion(self):
        if self.ss_epsilon > 0:
            return f"SS({self.ss_epsilon:.2f})"
        else:
            return super().criterion
        
    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith('ss'):
            return SoftLabelCrossEntropyLoss(**kwargs)
        else:
            return super().instantiate_criterion(**kwargs)
        
        
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
        if self.use_inv_rel: 
            self.idx2label = self.idx2label + [f"{INV_REL_PREFIX}{label}" for label in self.idx2label if label != self.none_label and label not in self.sym_rel_labels]
        
        counter = Counter((label, head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_ht_labels = set(rht[1:] for rht in self.existing_rht_labels)
        self.existing_self_rel = any(head[1:]==tail[1:] for data in partitions for entry in data for label, head, tail in entry['relations'])
        
        cp_distances = [chunk_pair_distance(head, tail) for data in partitions for entry in data for label, head, tail in entry['relations']]
        self.max_dist_id = math.ceil(numpy.quantile(cp_distances, MAX_DIST_ID_COV_RATE))
        self.max_cp_dist = max(cp_distances)
        
        
    def instantiate(self):
        return SpecificSpanRelClsDecoder(self)



class SpecificSpanRelClsDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: SpecificSpanRelClsDecoderConfig):
        super().__init__()
        self.use_context = config.use_context
        self.context_mode = config.context_mode
        self.fusing_mode = config.fusing_mode
        self.min_span_size = config.min_span_size
        self.max_span_size = config.max_span_size
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.ck_loss_weight = config.ck_loss_weight
        self.l2_loss_weight = config.l2_loss_weight
        self.sym_rel_labels = config.sym_rel_labels
        self.comp_sym_rel = config.comp_sym_rel
        self.use_inv_rel = config.use_inv_rel
        self.check_rht_labels = config.check_rht_labels
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.ss_epsilon = config.ss_epsilon
        self.existing_rht_labels = config.existing_rht_labels
        self.existing_ht_labels = config.existing_ht_labels
        self.existing_self_rel = config.existing_self_rel
        self.max_cp_dist = config.max_cp_dist
        self.max_dist_id = config.max_dist_id
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        
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
            if config.reduction_cat.num_layers > 0 and config.reduction_cat.out_dim > 0:
                self.reduction_cat = config.reduction_cat.instantiate()
                self.hid2logit = torch.nn.Linear(config.reduction_cat.out_dim, config.voc_dim)
            else:
                self.hid2logit = torch.nn.Linear(config.reduction_cat.in_dim, config.voc_dim)
            reinit_layer_(self.hid2logit, 'sigmoid')
            
        elif config.fusing_mode.lower().startswith('affine'):
            self.reduction_head = config.reduction.instantiate()
            self.reduction_tail = config.reduction.instantiate()
            if config.use_context: 
                self.reduction_ctx = config.reduction_ctx.instantiate()
                self.hid2logit = TriAffineFusor(config.reduction.out_dim, config.voc_dim)
            else:
                self.hid2logit = BiAffineFusor(config.reduction.out_dim, config.voc_dim)
        
        if config.ck_loss_weight > 0: 
            self.ck_hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim+config.label_emb_dim, config.ck_voc_dim)
            reinit_layer_(self.ck_hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for cp_obj, chunks_pred in zip(batch.cp_objs, batch_chunks_pred):
            if cp_obj.chunks_pred is None:
                cp_obj.chunks_pred = chunks_pred
                cp_obj.build(self)
                if self.fusing_mode.startswith('concat'):
                    cp_obj.to(self.hid2logit.weight.device)
                elif self.fusing_mode.lower().startswith('affine'):
                    cp_obj.to(self.hid2logit.U.device)
        
        
    def _collect_shallow_contexts(self, full_hidden: torch.Tensor, i: int, cp_obj: ChunkPairs): 
        # contexts: (num_chunks^2, ctx_span_size, hid_dim) -> (num_chunks^2, hid_dim) -> (num_chunks, num_chunks, hid_dim)
        contexts = []
        for (_, h_start, h_end), (_, t_start, t_end), is_valid in self.enumerate_chunk_pairs(cp_obj): 
            if not is_valid: 
                contexts.append(self.none_context.unsqueeze(0))
            elif h_end < t_start:
                contexts.append(full_hidden[i, h_end:t_start])
            elif t_end < h_start:
                contexts.append(full_hidden[i, t_end:h_start])
            else:
                contexts.append(self.zero_context.unsqueeze(0))
        ctx_mask = seq_lens2mask(torch.tensor([c.size(0) for c in contexts], dtype=torch.long, device=full_hidden.device))
        contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True, padding_value=0.0)
        contexts = self.aggregating(self.in_dropout(contexts), mask=ctx_mask)
        contexts = contexts.view(len(cp_obj.chunks), len(cp_obj.chunks), -1)
        return contexts 
        
        
    def _collect_specific_contexts(self, all_hidden: List[torch.Tensor], i: int, cp_obj: ChunkPairs):
        max_span_size = len(all_hidden)
        # contexts: (num_chunks^2, hid_dim) -> (num_chunks, num_chunks, hid_dim)
        contexts = []
        for (_, h_start, h_end), (_, t_start, t_end), is_valid in self.enumerate_chunk_pairs(cp_obj): 
            if not is_valid: 
                contexts.append(self.none_context)
            elif h_end < t_start or t_end < h_start:
                c_start, c_end = min(h_end, t_end), max(h_start, t_start)
                if c_end - c_start <= max_span_size:
                    curr_context = all_hidden[c_end-c_start-1][i, c_start]
                else:
                    curr_context = (all_hidden[max_span_size-1][i, c_start] + all_hidden[max_span_size-1][i, c_end-max_span_size]) / 2
                contexts.append(curr_context)
            else:
                contexts.append(self.zero_context)
        contexts = self.in_dropout(torch.stack(contexts))
        contexts = contexts.view(len(cp_obj.chunks), len(cp_obj.chunks), -1)
        return contexts
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        # full_hidden: (batch, step, hid_dim)
        # query_hidden: (batch, step-k+1, hid_dim)
        if self.min_span_size == 1: 
            all_hidden = list(all_query_hidden.values())
        else:
            all_hidden = [full_hidden] + list(all_query_hidden.values())
        
        batch_logits = []
        for i, cp_obj in enumerate(batch.cp_objs):
            if not cp_obj.has_valid_cp: 
                logits = None
            else: 
                # span_hidden: (num_chunks, hid_dim)
                span_hidden = torch.stack([all_hidden[end-start-1][i, start] for label, start, end in cp_obj.chunks])
                
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
                
                if self.fusing_mode.startswith('concat'):
                    # hidden_cat: (num_chunks, num_chunks, hid_dim*3)
                    hidden_cat = torch.cat([head_hidden, tail_hidden], dim=-1)
                    if self.use_context: 
                        if self.context_mode.lower().count('shallow'): 
                            contexts = self._collect_shallow_contexts(full_hidden, i, cp_obj)
                        else: 
                            contexts = self._collect_specific_contexts(all_hidden, i, cp_obj)
                        hidden_cat = torch.cat([hidden_cat, contexts], dim=-1)
                    if hasattr(self, 'reduction_cat'):
                        reduced_cat = self.reduction_cat(hidden_cat)
                        logits = self.hid2logit(self.hid_dropout(reduced_cat))
                    else: 
                        # logits: (num_chunks, num_chunks, logit_dim)
                        logits = self.hid2logit(hidden_cat)
                    
                elif self.fusing_mode.lower().startswith('affine'):
                    reduced_head = self.reduction_head(head_hidden)
                    reduced_tail = self.reduction_tail(tail_hidden)
                    if self.use_context: 
                        if self.context_mode.lower().count('shallow'): 
                            contexts = self._collect_shallow_contexts(full_hidden, i, cp_obj)
                        else: 
                            contexts = self._collect_specific_contexts(all_hidden, i, cp_obj)
                        reduced_ctx = self.reduction_ctx(contexts)
                        logits = self.hid2logit(self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail), self.hid_dropout(reduced_ctx))
                    else:
                        logits = self.hid2logit(self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail))
                
                if self.ck_loss_weight > 0:
                    ck_logits = self.ck_hid2logit(span_hidden)
                    logits = (logits, ck_logits)
            
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
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
                if self.l2_loss_weight > 0 and self.fusing_mode.lower().startswith('affine'):
                    loss = loss + self.l2_loss_weight*(self.hid2logit.U**2).sum()
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
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
