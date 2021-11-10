# -*- coding: utf-8 -*-
from typing import List, Tuple
from collections import Counter
import logging
import torch

from ...wrapper import TargetWrapper, Batch
from ...utils.chunk import detect_nested, filter_clashed_by_priority
from ...nn.modules import CombinedDropout, SoftLabelCrossEntropyLoss
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from ..encoder import EncoderConfig
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase

logger = logging.getLogger(__name__)


class BoundarySelectionDecoderMixin(DecoderMixinBase):
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


def _spans_from_surrounding(span: Tuple[int], distance: int, num_tokens: int):
    """Spans from the surrounding area of the given `span`.
    """
    for k in range(distance):
        for start_offset, end_offset in [(-k, -distance+k), 
                                         (-distance+k, k), 
                                         (k, distance-k), 
                                         (distance-k, -k)]:
            start, end = span[0]+start_offset, span[1]+end_offset
            if 0 <= start < end <= num_tokens:
                yield (start, end)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area. 
    """
    for start in range(seq_len):
        for end in range(start+1, seq_len+1):
            yield (start, end)


class Boundaries(TargetWrapper):
    """A wrapper of boundaries with underlying chunks. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    """
    def __init__(self, data_entry: dict, config: BoundarySelectionDecoderMixin, training: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry.get('chunks', None)
        num_tokens = len(data_entry['tokens'])
        
        if training and config.neg_sampling_rate < 1:
            non_mask = (torch.arange(num_tokens) - torch.arange(num_tokens).unsqueeze(-1) >= 0)
            pos_non_mask = torch.zeros_like(non_mask)
            for label, start, end in self.chunks:
                pos_non_mask[start, end-1] = True
            
            neg_sampled = torch.empty_like(non_mask).bernoulli(p=config.neg_sampling_rate)
            
            if config.hard_neg_sampling_rate > config.neg_sampling_rate:
                hard_neg_non_mask = torch.zeros_like(non_mask)
                for label, start, end in self.chunks:
                    for dist in range(1, config.hard_neg_sampling_size+1):
                        for sur_start, sur_end in _spans_from_surrounding((start, end), dist, num_tokens):
                            hard_neg_non_mask[sur_start, sur_end-1] = True
                
                if config.hard_neg_sampling_rate < 1:
                    # Solve: 1 - (1 - p_{neg})(1 - p_{comp}) = p_{hard}
                    # Get: p_{comp} = (p_{hard} - p_{neg}) / (1 - p_{neg})
                    comp_sampling_rate = (config.hard_neg_sampling_rate - config.neg_sampling_rate) / (1 - config.neg_sampling_rate)
                    comp_sampled = torch.empty_like(non_mask).bernoulli(p=comp_sampling_rate)
                    neg_sampled = neg_sampled | (comp_sampled & hard_neg_non_mask)
                else:
                    neg_sampled = neg_sampled | hard_neg_non_mask
            
            self.non_mask = pos_non_mask | (neg_sampled & non_mask)
        
        if self.chunks is not None:
            if config.sb_epsilon <= 0 and config.sl_epsilon <= 0:
                # Cross entropy loss
                self.boundary2label_id = torch.full((num_tokens, num_tokens), config.none_idx, dtype=torch.long)
                for label, start, end in self.chunks:
                    self.boundary2label_id[start, end-1] = config.label2idx[label]
            else:
                # Soft label loss for either boundary or label smoothing 
                self.boundary2label_id = torch.zeros(num_tokens, num_tokens, config.voc_dim, dtype=torch.float)
                for label, start, end in self.chunks:
                    label_id = config.label2idx[label]
                    self.boundary2label_id[start, end-1, label_id] += (1 - config.sb_epsilon)
                    
                    for dist in range(1, config.sb_size+1):
                        eps_per_span = config.sb_epsilon / (config.sb_size * dist * 4)
                        sur_spans = list(_spans_from_surrounding((start, end), dist, num_tokens))
                        for sur_start, sur_end in sur_spans:
                            self.boundary2label_id[sur_start, sur_end-1, label_id] += (eps_per_span*config.sb_adj_factor)
                        # Absorb the probabilities assigned to illegal positions
                        self.boundary2label_id[start, end-1, label_id] += eps_per_span * (dist * 4 - len(sur_spans))
                
                # In very rare cases (e.g., ACE 2005), multiple entities may have the same span but different types
                overflow_indic = (self.boundary2label_id.sum(dim=-1) > 1)
                if overflow_indic.any().item():
                    self.boundary2label_id[overflow_indic] = torch.nn.functional.normalize(self.boundary2label_id[overflow_indic], p=1, dim=-1)
                self.boundary2label_id[:, :, config.none_idx] = 1 - self.boundary2label_id.sum(dim=-1)
                
                if config.sl_epsilon > 0:
                    # Do not smooth to `<none>` label
                    pos_indic = (torch.arange(config.voc_dim) != config.none_idx)
                    self.boundary2label_id[:, :, pos_indic] = (self.boundary2label_id[:, :, pos_indic] * (1-config.sl_epsilon) + 
                                                               self.boundary2label_id[:, :, pos_indic].sum(dim=-1, keepdim=True)*config.sl_epsilon / (config.voc_dim-1))



class BoundarySelectionDecoderConfig(SingleDecoderConfigBase, BoundarySelectionDecoderMixin):
    def __init__(self, **kwargs):
        self.use_biaffine = kwargs.pop('use_biaffine', True)
        self.affine = kwargs.pop('affine', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2))
        
        self.max_len = kwargs.pop('max_len', None)
        self.max_span_size = kwargs.pop('max_span_size', 50)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        self.hard_neg_sampling_rate = kwargs.pop('hard_neg_sampling_rate', 1.0)
        self.hard_neg_sampling_rate = max(self.hard_neg_sampling_rate, self.neg_sampling_rate)
        self.hard_neg_sampling_size = kwargs.pop('hard_neg_sampling_size', 5)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        # Note: non-nested overlapping chunks are never allowed
        self.allow_nested = kwargs.pop('allow_nested', None)
        
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
        
        self.allow_nested = any(detect_nested(entry['chunks']) for data in partitions for entry in data)
        if self.allow_nested:
            logger.info("Nested chunks detected, nested chunks are allowed in decoding...")
        else:
            logger.info("No nested chunks detected, only flat chunks are allowed in decoding...")
        
        self.max_len = max(len(data_entry['tokens']) for data in partitions for data_entry in data)
        
        
    def instantiate(self):
        return BoundarySelectionDecoder(self)




class BoundarySelectionDecoder(DecoderBase, BoundarySelectionDecoderMixin):
    def __init__(self, config: BoundarySelectionDecoderConfig):
        super().__init__()
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.allow_nested = config.allow_nested
        
        if config.use_biaffine:
            self.affine_start = config.affine.instantiate()
            self.affine_end = config.affine.instantiate()
        else:
            self.affine = config.affine.instantiate()
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_span_size, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        # Use buffer to accelerate computation
        # Note: size_id = size - 1
        self.register_buffer('_span_size_ids', torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1))
        # Create `_span_non_mask` before changing values of `_span_size_ids`
        self.register_buffer('_span_non_mask', self._span_size_ids >= 0)
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids >= config.max_span_size, config.max_span_size-1)
        
        self.dropout = CombinedDropout(*config.hid_drop_rates)
        
        self.U = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim, config.affine.out_dim))
        self.W = torch.nn.Parameter(torch.empty(config.voc_dim, config.affine.out_dim*2 + config.size_emb_dim))
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
        if hasattr(self, 'affine_start'):
            affined_start = self.affine_start(full_hidden, batch.mask)
            affined_end = self.affine_end(full_hidden, batch.mask)
        else:
            affined_start = self.affine(full_hidden, batch.mask)
            affined_end = self.affine(full_hidden, batch.mask)
        
        # affined_start: (batch, start_step, affine_dim) -> (batch, 1, start_step, affine_dim)
        # affined_end: (batch, end_step, affine_dim) -> (batch, 1, affine_dim, end_step)
        # scores1: (batch, 1, start_step, affine_dim) * (voc_dim, affine_dim, affine_dim) * (batch, 1, affine_dim, end_step) -> (batch, voc_dim, start_step, end_step)
        scores1 = self.dropout(affined_start).unsqueeze(1).matmul(self.U).matmul(self.dropout(affined_end).permute(0, 2, 1).unsqueeze(1))
        
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
        return scores1.permute(0, 2, 3, 1) + scores2.squeeze(-1) + self.b
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden)
        
        losses = []
        for curr_scores, boundaries_obj, curr_len in zip(batch_scores, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            curr_non_mask = getattr(boundaries_obj, 'non_mask', self._get_span_non_mask(curr_len))
            
            loss = self.criterion(curr_scores[:curr_len, :curr_len][curr_non_mask], boundaries_obj.boundary2label_id[curr_non_mask])
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_scores = self.compute_scores(batch, full_hidden)
        
        batch_chunks = []
        for curr_scores, curr_len in zip(batch_scores, batch.seq_lens.cpu().tolist()):
            curr_non_mask = self._get_span_non_mask(curr_len)
            
            confidences, label_ids = curr_scores[:curr_len, :curr_len][curr_non_mask].softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
            chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_upper_triangular(curr_len)) if label != self.none_label]
            confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            assert len(confidences) == len(chunks)
            
            # Sort chunks from high to low confidences
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_nested=self.allow_nested)
            
            batch_chunks.append(chunks)
        return batch_chunks
