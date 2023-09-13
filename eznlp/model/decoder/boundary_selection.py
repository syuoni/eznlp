# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...utils.chunk import detect_overlapping_level, filter_clashed_by_priority
from ...nn.modules import CombinedDropout, SoftLabelCrossEntropyLoss
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from ..encoder import EncoderConfig
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .boundaries import Boundaries, MAX_SIZE_ID_COV_RATE, _spans_from_upper_triangular

logger = logging.getLogger(__name__)


class BoundariesDecoderMixin(DecoderMixinBase):
    """The standard `Mixin` for span-based entity recognition. 
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
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return {'boundaries_obj': Boundaries(data_entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'boundaries_objs': [ex['boundaries_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [boundaries_obj.chunks for boundaries_obj in batch.boundaries_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        """Micro-F1 for entity recognition. 
        
        References
        ----------
        https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        """
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']
        
        
    def _filter(self, chunks: List[tuple], confidences: List[float], boundaries_obj):
        if hasattr(boundaries_obj, 'sub2ori_idx'):
            is_valid = [isinstance(boundaries_obj.sub2ori_idx[start], int) and isinstance(boundaries_obj.sub2ori_idx[end], int) for label, start, end in chunks]
            confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
            chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]
        
        if hasattr(boundaries_obj, 'tok2sent_idx'):
            is_valid = [boundaries_obj.tok2sent_idx[start] == boundaries_obj.tok2sent_idx[end-1] for label, start, end in chunks]
            confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
            chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]
        
        if self.chunk_priority.lower().startswith('len'):
            # Sort chunks by lengths: long -> short 
            chunks = sorted(chunks, key=lambda ck: ck[2]-ck[1], reverse=True)
        else:
            # Sort chunks by confidences: high -> low 
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
        chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)
        return chunks



class BoundarySelectionDecoderConfig(SingleDecoderConfigBase, BoundariesDecoderMixin):
    def __init__(self, **kwargs):
        self.reduction = kwargs.pop('reduction', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0))
        
        self.max_len = kwargs.pop('max_len', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        self.neg_sampling_power_decay = kwargs.pop('neg_sampling_power_decay', 0.0)  # decay = 0.5, 1.0
        self.neg_sampling_surr_rate = kwargs.pop('neg_sampling_surr_rate', 0.0)
        self.neg_sampling_surr_size = kwargs.pop('neg_sampling_surr_size', 5)
        self.nested_sampling_rate = kwargs.pop('nested_sampling_rate', 1.0)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.overlapping_level = kwargs.pop('overlapping_level', None)
        self.chunk_priority = kwargs.pop('chunk_priority', 'confidence')
        
        # Boundary smoothing epsilon
        self.sb_epsilon = kwargs.pop('sb_epsilon', 0.0)
        self.sb_size = kwargs.pop('sb_size', 1)
        self.sb_adj_factor = kwargs.pop('sb_adj_factor', 1.0)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.reduction.arch, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'hid_drop_rates', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def in_dim(self):
        return self.reduction.in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        self.reduction.in_dim = dim
        
    @property
    def criterion(self):
        if self.sb_epsilon > 0:
            crit_name = f"SB({self.sb_epsilon:.2f}, {self.sb_size})"
            return f"B{crit_name}" if self.multilabel else crit_name
        else:
            return super().criterion
        
    def instantiate_criterion(self, **kwargs):
        if self.multilabel:
            # `BCEWithLogitsLoss` allows the target to be any continuous value in [0, 1]
            return torch.nn.BCEWithLogitsLoss(**kwargs)
        elif self.criterion.lower().startswith(('sb', 'sl')):
            # For boundary/label smoothing, the `Boundaries` object has been accordingly changed; 
            # hence, do not use `SmoothLabelCrossEntropyLoss`
            return SoftLabelCrossEntropyLoss(**kwargs)
        else:
            return super().instantiate_criterion(**kwargs)
        
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        self.overlapping_level = max(detect_overlapping_level(entry['chunks']) for data in partitions for entry in data)
        logger.info(f"Overlapping level: {self.overlapping_level}")
        
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        self.max_size_id = math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        
        self.max_len = max(len(data_entry['tokens']) for data in partitions for data_entry in data)
        
        
    def instantiate(self):
        return BoundarySelectionDecoder(self)




class BoundarySelectionDecoder(DecoderBase, BoundariesDecoderMixin):
    def __init__(self, config: BoundarySelectionDecoderConfig):
        super().__init__()
        self.multilabel = config.multilabel
        self.conf_thresh = config.conf_thresh
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.overlapping_level = config.overlapping_level
        self.chunk_priority = config.chunk_priority
        
        self.reduction_start = config.reduction.instantiate()
        self.reduction_end = config.reduction.instantiate()
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.register_buffer('_span_size_ids', torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1))
        # Create `_span_non_mask` before changing values of `_span_size_ids`
        self.register_buffer('_span_non_mask', self._span_size_ids >= 0)
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids > config.max_size_id, config.max_size_id)
        
        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)
        
        self.U = torch.nn.Parameter(torch.empty(config.voc_dim, config.reduction.out_dim, config.reduction.out_dim))
        self.W = torch.nn.Parameter(torch.empty(config.voc_dim, config.reduction.out_dim*2 + config.size_emb_dim))
        self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
        torch.nn.init.orthogonal_(self.U.data)
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def _get_span_size_ids(self, seq_len: int):
        return self._span_size_ids[:seq_len, :seq_len]
        
    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]
        
        
    def compute_scores(self, batch: Batch, full_hidden: torch.Tensor):
        reduced_start = self.reduction_start(self.in_dropout(full_hidden), batch.mask)
        reduced_end = self.reduction_end(self.in_dropout(full_hidden), batch.mask)
        
        # reduced_start: (batch, start_step, red_dim) -> (batch, 1, start_step, red_dim)
        # reduced_end: (batch, end_step, red_dim) -> (batch, 1, red_dim, end_step)
        # scores1: (batch, 1, start_step, red_dim) * (voc_dim, red_dim, red_dim) * (batch, 1, red_dim, end_step) -> (batch, voc_dim, start_step, end_step)
        scores1 = self.hid_dropout(reduced_start).unsqueeze(1).matmul(self.U).matmul(self.hid_dropout(reduced_end).permute(0, 2, 1).unsqueeze(1))
        
        # reduced_cat: (batch, start_step, end_step, red_dim*2)
        reduced_cat = torch.cat([self.hid_dropout(reduced_start).unsqueeze(2).expand(-1, -1, reduced_end.size(1), -1), 
                                 self.hid_dropout(reduced_end).unsqueeze(1).expand(-1, reduced_start.size(1), -1, -1)], dim=-1)
        
        if hasattr(self, 'size_embedding'):
            # size_embedded: (start_step, end_step, emb_dim)
            size_embedded = self.size_embedding(self._get_span_size_ids(full_hidden.size(1)))
            # reduced_cat: (batch, start_step, end_step, red_dim*2 + emb_dim)
            reduced_cat = torch.cat([reduced_cat, self.hid_dropout(size_embedded).unsqueeze(0).expand(full_hidden.size(0), -1, -1, -1)], dim=-1)
        
        # scores2: (voc_dim, red_dim*2 + emb_dim) * (batch, start_step, end_step, red_dim*2 + emb_dim, 1) -> (batch, start_step, end_step, voc_dim, 1)
        scores2 = self.W.matmul(reduced_cat.unsqueeze(-1))
        
        # scores: (batch, start_step, end_step, voc_dim)
        scores = scores1.permute(0, 2, 3, 1) + scores2.squeeze(-1)
        return scores + self.b
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden)
        
        losses = []
        for curr_scores, boundaries_obj, curr_len in zip(batch_scores, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            curr_non_mask = getattr(boundaries_obj, 'non_mask', self._get_span_non_mask(curr_len))
            
            loss = self.criterion(curr_scores[:curr_len, :curr_len][curr_non_mask], boundaries_obj.label_ids[curr_non_mask])
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden)
        
        batch_chunks = []
        for curr_scores, boundaries_obj, curr_len in zip(batch_scores, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            curr_non_mask = self._get_span_non_mask(curr_len)
            
            if not self.multilabel:
                confidences, label_ids = curr_scores[:curr_len, :curr_len][curr_non_mask].softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
                chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != self.none_label]
                confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            else:
                all_confidences = curr_scores[:curr_len, :curr_len][curr_non_mask].sigmoid()
                # Zero-out all spans according to <none> labels
                all_confidences[all_confidences[:,self.none_idx] > (1-self.conf_thresh)] = 0
                # Zero-out <none> labels for all spans
                all_confidences[:,self.none_idx] = 0
                all_spans = list(_spans_from_upper_triangular(curr_len))
                assert all_confidences.size(0) == len(all_spans)
                
                all_confidences_list = all_confidences.cpu().tolist()
                pos_entries = torch.nonzero(all_confidences > self.conf_thresh).cpu().tolist()
                # In the early training stage, the chunk-decoder may produce too many predicted chunks
                MAX_NUM_CHUNKS = 500
                if len(pos_entries) > MAX_NUM_CHUNKS:
                    pos_entries = pos_entries[:MAX_NUM_CHUNKS]
                
                chunks = [(self.idx2label[i], *all_spans[sidx]) for sidx, i in pos_entries]
                confidences = [all_confidences_list[sidx][i] for sidx, i in pos_entries]
            
            assert len(confidences) == len(chunks)
            chunks = self._filter(chunks, confidences, boundaries_obj)
            batch_chunks.append(chunks)
        return batch_chunks
