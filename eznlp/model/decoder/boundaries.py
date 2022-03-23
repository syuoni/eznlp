from typing import Tuple
import random
import torch

from ...wrapper import TargetWrapper
from .base import SingleDecoderConfigBase


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


def _spans_from_diagonals(seq_len: int, max_span_size: int=None):
    """Spans from diagonals of the upper triangular area. 
    """
    if max_span_size is None:
        max_span_size = seq_len
    
    for k in range(1, max_span_size+1):
        for start in range(seq_len-k+1):
            yield (start, start+k)



class Boundaries(TargetWrapper):
    """A wrapper of boundaries with underlying chunks. 
    
    Eberts and Ulges (2019) use a fixed number of negative samples as 100. 
    Li et al. (2021) recommend negative sampling rate as 0.3 to 0.4. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    
    References
    ----------
    [1] Eberts and Ulges. 2019. Span-based joint entity and relation extraction with Transformer pre-training. ECAI 2020.
    [2] Li et al. 2021. Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition. ICLR 2021. 
    """
    def __init__(self, data_entry: dict, config: SingleDecoderConfigBase, training: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry.get('chunks', None)
        num_tokens = len(data_entry['tokens'])
        
        if training and (config.neg_sampling_rate < 1 or config.neg_sampling_power_decay > 0):
            span_size_ids = torch.arange(num_tokens) - torch.arange(num_tokens).unsqueeze(-1)
            non_mask_rate = config.neg_sampling_rate * (span_size_ids+1)**(-config.neg_sampling_power_decay)
            non_mask_rate.masked_fill_(span_size_ids < 0, 0)
            non_mask_rate.clamp_(max=1)
            
            # Sampling rate is 1 for positive samples
            for label, start, end in self.chunks:
                non_mask_rate[start, end-1] = 1
            
            # Extra sampling rate surrounding positive samples
            if config.neg_sampling_surr_rate > 0 and config.neg_sampling_surr_size > 0:
                surr_non_mask = torch.zeros_like(non_mask_rate, dtype=torch.bool)
                for label, start, end in self.chunks:
                    for dist in range(1, config.neg_sampling_surr_size+1):
                        for surr_start, surr_end in _spans_from_surrounding((start, end), dist, num_tokens):
                            surr_non_mask[surr_start, surr_end-1] = True
                non_mask_rate[surr_non_mask] += (1 - non_mask_rate[surr_non_mask]) * config.neg_sampling_surr_rate
            
            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool() 
            
            # In case of all masked (may appears when no positive samples, very short sequence, and low negative sampling rate), 
            # randomly re-select one span of size 1 for un-masking. 
            if not self.non_mask.any().item():
                start = random.randrange(num_tokens)
                self.non_mask[start, start] = True
        
        if self.chunks is not None:
            if config.sb_epsilon <= 0 and config.sl_epsilon <= 0:
                # Cross entropy loss for non-smoothing
                self.boundary2label_id = torch.full((num_tokens, num_tokens), config.none_idx, dtype=torch.long)
                for label, start, end in self.chunks:
                    self.boundary2label_id[start, end-1] = config.label2idx[label]
            else:
                # Soft label loss for boundary/label smoothing 
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
                
                # In very rare cases of some datasets (e.g., ACE 2005), multiple entities may have the same span but different types
                overflow_indic = (self.boundary2label_id.sum(dim=-1) > 1)
                if overflow_indic.any().item():
                    self.boundary2label_id[overflow_indic] = torch.nn.functional.normalize(self.boundary2label_id[overflow_indic], p=1, dim=-1)
                self.boundary2label_id[:, :, config.none_idx] = 1 - self.boundary2label_id.sum(dim=-1)
                
                if config.sl_epsilon > 0:
                    # Do not smooth to `<none>` label
                    pos_indic = (torch.arange(config.voc_dim) != config.none_idx)
                    self.boundary2label_id[:, :, pos_indic] = (self.boundary2label_id[:, :, pos_indic] * (1-config.sl_epsilon) + 
                                                               self.boundary2label_id[:, :, pos_indic].sum(dim=-1, keepdim=True)*config.sl_epsilon / (config.voc_dim-1))
        
        
    def diagonal_label_ids(self, max_span_size: int=None):
        if max_span_size is None:
            max_span_size = self.boundary2label_id.size(0)
        
        # label_ids: (\sum_k seq_len-k+1, )
        label_ids = torch.cat([self.boundary2label_id.diagonal(offset=k-1) for k in range(1, max_span_size+1)], dim=-1)
        if label_ids.dim() == 2:
            # label_ids: (\sum_k seq_len-k+1, logit_dim)
            label_ids = label_ids.permute(1, 0)
        return label_ids
        
        
    def diagonal_non_mask(self, max_span_size: int=None):
        if max_span_size is None:
            max_span_size = self.boundary2label_id.size(0)
        
        # non_mask: (\sum_k seq_len-k+1, )
        return torch.cat([self.non_mask.diagonal(offset=k-1) for k in range(1, max_span_size+1)], dim=-1)
