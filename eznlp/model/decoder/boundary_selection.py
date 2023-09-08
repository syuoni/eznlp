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



class BoundarySelectionDecoderConfig(SingleDecoderConfigBase, BoundariesDecoderMixin):
    def __init__(self, **kwargs):
        self.use_biaffine = kwargs.pop('use_biaffine', True)
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        self.use_prod = kwargs.pop('use_prod', True)
        
        self.max_len = kwargs.pop('max_len', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
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
        return self._name_sep.join([self.affine.arch, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'hid_drop_rates', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def in_dim(self):
        return self.affine.in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        self.affine.in_dim = dim
        
    @property
    def criterion(self):
        if self.sb_epsilon > 0:
            return f"SB({self.sb_epsilon:.2f}, {self.sb_size})"
        else:
            return super().criterion
        
    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith(('sb', 'sl')):
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
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.overlapping_level = config.overlapping_level
        self.chunk_priority = config.chunk_priority
        
        if config.use_biaffine:
            self.affine_start = config.affine.instantiate()
            self.affine_end = config.affine.instantiate()
        else:
            self.affine = config.affine.instantiate()
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.register_buffer('_span_size_ids', torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1))
        # Create `_span_non_mask` before changing values of `_span_size_ids`
        self.register_buffer('_span_non_mask', self._span_size_ids >= 0)
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids > config.max_size_id, config.max_size_id)
        
        self.dropout = CombinedDropout(*config.hid_drop_rates)
        
        if config.use_prod:
            self.U = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim))
            torch.nn.init.orthogonal_(self.U.data)
        
        self.W = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim*2 + config.size_emb_dim))
        self.b = torch.nn.Parameter(torch.empty(config.voc_dim))
        # TODO: Check the output std.dev. 
        torch.nn.init.orthogonal_(self.W.data)
        torch.nn.init.zeros_(self.b.data)
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def _get_span_size_ids(self, seq_len: int):
        return self._span_size_ids[:seq_len, :seq_len]
        
    def _get_span_non_mask(self, seq_len: int):
        return self._span_non_mask[:seq_len, :seq_len]
        
        
    def compute_scores(self, batch: Batch, full_hidden: torch.Tensor):
        if hasattr(self, 'affine_start'):
            affined_start = self.affine_start(full_hidden, batch.mask)
            affined_end = self.affine_end(full_hidden, batch.mask)
        else:
            affined_start = self.affine(full_hidden, batch.mask)
            affined_end = self.affine(full_hidden, batch.mask)
        
        if hasattr(self, 'U'):
            # affined_start: (batch, start_step, affine_dim) -> (batch, 1, start_step, affine_dim)
            # affined_end: (batch, end_step, affine_dim) -> (batch, 1, affine_dim, end_step)
            # scores1: (batch, 1, start_step, affine_dim) * (voc_dim, affine_dim, affine_dim) * (batch, 1, affine_dim, end_step) -> (batch, voc_dim, start_step, end_step)
            scores1 = self.dropout(affined_start).unsqueeze(1).matmul(self.U).matmul(self.dropout(affined_end).permute(0, 2, 1).unsqueeze(1))
            # scores: (batch, start_step, end_step, voc_dim)
            scores = scores1.permute(0, 2, 3, 1)
        else:
            scores = 0
        
        # affined_cat: (batch, start_step, end_step, affine_dim*2)
        affined_cat = torch.cat([self.dropout(affined_start).unsqueeze(2).expand(-1, -1, affined_end.size(1), -1), 
                                 self.dropout(affined_end).unsqueeze(1).expand(-1, affined_start.size(1), -1, -1)], dim=-1)
        
        if hasattr(self, 'size_embedding'):
            # size_embedded: (start_step, end_step, emb_dim)
            size_embedded = self.size_embedding(self._get_span_size_ids(full_hidden.size(1)))
            # affined_cat: (batch, start_step, end_step, affine_dim*2 + emb_dim)
            affined_cat = torch.cat([affined_cat, self.dropout(size_embedded).unsqueeze(0).expand(full_hidden.size(0), -1, -1, -1)], dim=-1)
        
        # scores2: (voc_dim, affine_dim*2 + emb_dim) * (batch, start_step, end_step, affine_dim*2 + emb_dim, 1) -> (batch, start_step, end_step, voc_dim, 1)
        scores2 = self.W.matmul(affined_cat.unsqueeze(-1))
        # scores: (batch, start_step, end_step, voc_dim)
        scores = scores + scores2.squeeze(-1)
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
            
            confidences, label_ids = curr_scores[:curr_len, :curr_len][curr_non_mask].softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
            chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != self.none_label]
            confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            assert len(confidences) == len(chunks)
            
            if hasattr(boundaries_obj, 'sub2ori_idx'):
                is_valid = [isinstance(boundaries_obj.sub2ori_idx[start], int) and isinstance(boundaries_obj.sub2ori_idx[end], int) for label, start, end in chunks]
                confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
                chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]
            
            if self.chunk_priority.lower().startswith('len'):
                # Sort chunks by lengths: long -> short 
                chunks = sorted(chunks, key=lambda ck: ck[2]-ck[1], reverse=True)
            else:
                # Sort chunks by confidences: high -> low 
                chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)
            
            batch_chunks.append(chunks)
        return batch_chunks
