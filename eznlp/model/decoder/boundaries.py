from typing import Tuple
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
    
    Li et al. (2021) recommend negative sampling rate as 0.3 to 0.4. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    
    References
    ----------
    [1] Li et al. Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition. ICLR 2021. 
    """
    def __init__(self, data_entry: dict, config: SingleDecoderConfigBase, training: bool=True):
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
