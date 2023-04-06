# -*- coding: utf-8 -*-
from typing import Union, Tuple, List
import itertools
import random
import math
import torch

from ...wrapper import TargetWrapper
from .base import SingleDecoderConfigBase, DecoderBase


MAX_SIZE_ID_COV_RATE = 0.975


def _spans_from_surrounding(span: Tuple[int], distance: int, num_tokens: int):
    """Spans with given `distance` to the given `span`.
    """
    for k in range(distance):
        for start_offset, end_offset in [(-k, -distance+k), 
                                         (-distance+k, k), 
                                         (k, distance-k), 
                                         (distance-k, -k)]:
            start, end = span[0]+start_offset, span[1]+end_offset
            if 0 <= start < end <= num_tokens:
                yield (start, end)


def _spans_from_nested(span: Tuple[int]):
    for start in range(span[0], span[1]):
        for end in range(start+1, span[1]+1):
            if (start, end) != span:
                yield (start, end)


def _spans_from_upper_triangular(seq_len: int):
    """Spans from the upper triangular area. 
    """
    for start in range(seq_len):
        for end in range(start+1, seq_len+1):
            yield (start, end)


def _spans_from_diagonals(seq_len: int, max_span_size: int=None):
    """Spans from diagonals of the upper triangular area. 
    """
    if max_span_size is None or max_span_size > seq_len:
        max_span_size = seq_len
    
    for k in range(1, max_span_size+1):
        for start in range(seq_len-k+1):
            yield (start, start+k)


def _span_sizes_from_diagonals(seq_len: int, max_span_size: int=None):
    if max_span_size is None or max_span_size > seq_len:
        max_span_size = seq_len
    
    for k in range(1, max_span_size+1):
        for start in range(seq_len-k+1):
            yield k


def _ij2diagonal(i: int, j: int, seq_len: int):
    assert i <= j
    return (seq_len*2 - (j-i-1)) * (j-i) // 2 + i


def _diagonal2ij(k: int, seq_len: int):
    assert k < (seq_len+1)*seq_len // 2
    j_minus_i = int((2*seq_len+1 - math.sqrt((2*seq_len+1)**2 - 8*k)) / 2)
    i = k - ((seq_len*2 - (j_minus_i-1)) * j_minus_i // 2)
    return (i, i+j_minus_i)


def _span2diagonal(start: int, end: int, seq_len: int):
    return _ij2diagonal(start, end-1, seq_len)


def _diagonal2span(k: int, seq_len: int):
    i, j = _diagonal2ij(k, seq_len)
    return (i, j+1)


def _span_pairs_from_diagonals(seq_len: int, max_span_size: int=None):
    yield from itertools.product(_spans_from_diagonals(seq_len, max_span_size=max_span_size), 
                                 _spans_from_diagonals(seq_len, max_span_size=max_span_size))



class Boundaries(TargetWrapper):
    """A wrapper of boundaries with underlying chunks. 
    
    Eberts and Ulges (2019) use a fixed number of negative samples as 100. 
    Li et al. (2021) recommend negative sampling rate as 0.3 to 0.4. 
    
    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    
    References
    ----------
    [1] Eberts and Ulges. 2019. Span-based joint entity and relation extraction with Transformer pre-training. ECAI 2020.
    [2] Li et al. 2021. Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition. ICLR 2021. 
    """
    def __init__(self, entry: dict, config: SingleDecoderConfigBase, training: bool=True):
        super().__init__(training)
        
        self.chunks = entry.get('chunks', None)
        if 'sub2ori_idx' in entry:
            self.sub2ori_idx = entry['sub2ori_idx']
            self.ori2sub_idx = entry['ori2sub_idx']
        if 'tok2sent_idx' in entry:
            self.tok2sent_idx = entry['tok2sent_idx']
        
        self.num_tokens = len(entry['tokens'])
        self.max_span_size = min(getattr(config, 'max_span_size', self.num_tokens), self.num_tokens)
        
        if getattr(config, 'inex_mkmmd_lambda', 0.0) > 0 or config.nested_sampling_rate < 1:
            self.nest_non_mask = torch.zeros(self.num_tokens, self.num_tokens, dtype=torch.bool)
            for label, start, end in self.chunks:
                for nest_start, nest_end in _spans_from_nested((start, end)):
                    self.nest_non_mask[nest_start, nest_end-1] = True
        
        if training and (config.neg_sampling_rate < 1 or 
                         config.neg_sampling_power_decay > 0 or 
                         config.nested_sampling_rate < 1):
            span_size_ids = torch.arange(self.num_tokens) - torch.arange(self.num_tokens).unsqueeze(-1)
            non_mask_rate = config.neg_sampling_rate * (span_size_ids+1)**(-config.neg_sampling_power_decay)
            non_mask_rate.masked_fill_(span_size_ids < 0, 0)
            non_mask_rate.clamp_(max=1)
            
            # Extra sampling rate surrounding positive samples
            # p <- p + (1-p)*p_{surr}
            if config.neg_sampling_surr_rate > 0 and config.neg_sampling_surr_size > 0:
                surr_non_mask = torch.zeros_like(non_mask_rate, dtype=torch.bool)
                for label, start, end in self.chunks:
                    for dist in range(1, config.neg_sampling_surr_size+1):
                        for surr_start, surr_end in _spans_from_surrounding((start, end), dist, self.num_tokens):
                            surr_non_mask[surr_start, surr_end-1] = True
                non_mask_rate[surr_non_mask] += (1 - non_mask_rate[surr_non_mask]) * config.neg_sampling_surr_rate
            
            # Reduce sampling rate for spans nested in positive samples
            # p <- p * p_{nest}
            if config.nested_sampling_rate < 1:
                non_mask_rate[self.nest_non_mask] *= config.nested_sampling_rate
            
            # Sampling rate set to 1 for positive samples
            for label, start, end in self.chunks:
                non_mask_rate[start, end-1] = 1
            
            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool() 
            
            # In case of all masked (may appears when no positive samples, very short sequence, and low negative sampling rate), 
            # randomly re-select one span of size 1 for un-masking. 
            if not self.non_mask.any().item():
                start = random.randrange(self.num_tokens)
                self.non_mask[start, start] = True
        
        if self.chunks is not None:
            if config.sb_epsilon <= 0 and config.sl_epsilon <= 0:
                # Cross entropy loss for non-smoothing
                self.label_ids = torch.full((self.num_tokens, self.num_tokens), config.none_idx, dtype=torch.long)
                for label, start, end in self.chunks:
                    self.label_ids[start, end-1] = config.label2idx[label]
            else:
                # Soft label loss for boundary/label smoothing 
                self.label_ids = torch.zeros(self.num_tokens, self.num_tokens, config.voc_dim, dtype=torch.float)
                for label, start, end in self.chunks:
                    label_id = config.label2idx[label]
                    self.label_ids[start, end-1, label_id] += (1 - config.sb_epsilon)
                    
                    for dist in range(1, config.sb_size+1):
                        eps_per_span = config.sb_epsilon / (config.sb_size * dist * 4)
                        sur_spans = list(_spans_from_surrounding((start, end), dist, self.num_tokens))
                        for sur_start, sur_end in sur_spans:
                            self.label_ids[sur_start, sur_end-1, label_id] += (eps_per_span*config.sb_adj_factor)
                        # Absorb the probabilities assigned to illegal positions
                        self.label_ids[start, end-1, label_id] += eps_per_span * (dist * 4 - len(sur_spans))
                
                # In very rare cases of some datasets (e.g., ACE 2005), multiple entities may have the same span but different types
                overflow_indic = (self.label_ids.sum(dim=-1) > 1)
                if overflow_indic.any().item():
                    self.label_ids[overflow_indic] = torch.nn.functional.normalize(self.label_ids[overflow_indic], p=1, dim=-1)
                self.label_ids[:, :, config.none_idx] = 1 - self.label_ids.sum(dim=-1)
                
                if config.sl_epsilon > 0:
                    # Do not smooth to `<none>` label
                    pos_indic = (torch.arange(config.voc_dim) != config.none_idx)
                    self.label_ids[:, :, pos_indic] = (self.label_ids[:, :, pos_indic] * (1-config.sl_epsilon) + 
                                                       self.label_ids[:, :, pos_indic].sum(dim=-1, keepdim=True)*config.sl_epsilon / (config.voc_dim-1))
        
        
    def __getattr__(self, name):
        if name.startswith('diagonal_'):
            # x: (seq_len, seq_len)
            x = getattr(self, name.replace('diagonal_', ''))
            # diagonal_x: (\sum_k seq_len-k+1, )
            diagonal_x = torch.cat([x.diagonal(offset=k-1) for k in range(1, self.max_span_size+1)], dim=-1)
            if diagonal_x.dim() == 2:
                # diagonal_x: (\sum_k seq_len-k+1, logit_dim)
                diagonal_x = diagonal_x.permute(1, 0)
            return diagonal_x
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")



class DiagBoundariesPairs(TargetWrapper):
    """A wrapper of boundaries-pairs (in the diagonal format) with underlying relations. 
    This object enumerates all pairs between all possible spans (not exceeding `max_span_size`), 
    regardless of the spans being positive samples (entities) or not. Hence, it is not recommended 
    to use auxiliary chunk-type embeddings. 
    
    For pipeline modeling, `chunks_pred` is pre-computed, and thus initially non-empty in `entry`;
    For joint modeling, `chunks_pred` is computed on-the-fly, and thus initially empty in `entry`.
    
    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple], 
         'relations': List[tuple]}
    """
    def __init__(self, entry: dict, config: SingleDecoderConfigBase, training: bool=True):
        super().__init__(training)
        
        self.num_tokens = len(entry['tokens'])
        self.max_span_size = min(getattr(config, 'max_span_size', self.num_tokens), self.num_tokens)
        
        self.chunks_gold = entry['chunks'] if training else []
        self.chunks_pred = entry.get('chunks_pred', None)
        self.relations = entry.get('relations', None)
        if self.chunks_pred is not None:
            self.build(config)
        
    @property
    def chunks_pred(self):
        return self._chunks_pred
        
    @chunks_pred.setter
    def chunks_pred(self, chunks: List[tuple]):
        # `chunks_pred` is unchangable once set
        # In the evaluation phase, the outside chunk-decoder should produce deterministic predicted chunks
        assert getattr(self, '_chunks_pred', None) is None
        self._chunks_pred = chunks
        
        if self.chunks_pred is not None:
            # Do not use ```chunks = list(set(chunks_gold + chunks_pred))```, which may return non-deterministic order. 
            self.chunks = self.chunks_gold + [ck for ck in self.chunks_pred if ck not in self.chunks_gold]
            
            # In case of one span with multiple labels, the latter chunks will override the former ones; 
            # hence, the labels from `chunks_pred` are of priority. 
            # This only affects the training phase, because `chunks_gold` is always empty for evaluation phase. 
            self.span2ck_label = {(start, end): label for label, start, end in self.chunks}
        
        
    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_spans = (self.num_tokens*2 - (self.max_span_size-1)) * self.max_span_size // 2
        
        if self.training and config.neg_sampling_rate < 1:
            non_mask_rate = config.neg_sampling_rate * torch.ones(num_spans, num_spans, dtype=torch.float)
            for label, (_, h_start, h_end), (_, t_start, t_end) in self.relations:
                if h_end - h_start <= self.max_span_size and t_end - t_start <= self.max_span_size:
                    hk = _span2diagonal(h_start, h_end, self.num_tokens)
                    tk = _span2diagonal(t_start, t_end, self.num_tokens)
                    non_mask_rate[hk, tk] = 1
            
            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool() 
        
        if self.relations is not None:
            self.dbp2label_id = torch.full((num_spans, num_spans), config.none_idx, dtype=torch.long)
            for label, (_, h_start, h_end), (_, t_start, t_end) in self.relations:
                if h_end - h_start <= self.max_span_size and t_end - t_start <= self.max_span_size:
                    hk = _span2diagonal(h_start, h_end, self.num_tokens)
                    tk = _span2diagonal(t_start, t_end, self.num_tokens)
                    self.dbp2label_id[hk, tk] = config.label2idx[label]
