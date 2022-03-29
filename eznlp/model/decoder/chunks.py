# -*- coding: utf-8 -*-
from typing import List, Union
import torch

from ...wrapper import TargetWrapper
from .base import SingleDecoderConfigBase, DecoderBase


class ChunkPairs(TargetWrapper):
    """A wrapper of chunk-pairs with underlying relations. 
    This object enumerates all pairs between all positive spans (entity spans). 
    
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
        assert getattr(self, '_chunks_pred', None) is None  # `chunks_pred` is unchangable once set
        self._chunks_pred = chunks
        
        if self.chunks_pred is not None:
            # Do not use ```self.chunks = list(set(self.chunks_gold + self.chunks_pred))```,
            # which may return non-deterministic order. 
            self.chunks = self.chunks_gold + [ck for ck in self.chunks_pred if ck not in self.chunks_gold]
            
            # In merged chunks, there may exist two chunks with a same span but different chunk-types. 
            # In this case, auxiliary chunk-type embeddings may be useful. 
            self.chunk2idx = {ck: i for i, ck in enumerate(self.chunks)}
        
        
    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_chunks = len(self.chunks)
        
        self.span_size_ids = torch.tensor([end-start-1 for label, start, end in self.chunks], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        self.ck_label_ids = torch.tensor([config.ck_label2idx[label] for label, start, end in self.chunks], dtype=torch.long)
        
        if self.training and config.neg_sampling_rate < 1:
            non_mask_rate = config.neg_sampling_rate * torch.ones(num_chunks, num_chunks, dtype=torch.float)
            for label, head, tail in self.relations:
                hk = self.chunk2idx[head]
                tk = self.chunk2idx[tail]
                non_mask_rate[hk, tk] = 1
            
            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool() 
        
        if self.relations is not None:
            self.cp2label_id = torch.full((num_chunks, num_chunks), config.none_idx, dtype=torch.long)
            for label, head, tail in self.relations:
                # `head`/`tail` may not appear in `chunks` in the evaluation phase where `chunks_gold` are not allowed to access
                if (not self.training) and (head not in self.chunk2idx or tail not in self.chunk2idx):
                    continue
                hk = self.chunk2idx[head]
                tk = self.chunk2idx[tail]
                self.cp2label_id[hk, tk] = config.label2idx[label]



class ChunkSingles(TargetWrapper):
    """A wrapper of chunks with underlying attributes. 
    
    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple], 
         'attributes': List[tuple]}
    
    Notes
    -----
    (1) If `building` is `True`, `entry['chunks']` is assumed to be known for both training and evaluation (e.g., pipeline modeling).
        In this case, the negative samples are generated from `entry['chunks']`. 
    (2) If `building` is `False`, `entry['chunks']` is known for training but not for evaluation (e.g., joint modeling).
        In this case, `inject_chunks` and `build` should be successively invoked, and the negative samples are generated from injected chunks. 
    """
    def __init__(self, entry: dict, config: SingleDecoderConfigBase, training: bool=True, building: bool=True):
        super().__init__(training)
        
        self.chunks = entry['chunks'] if training or building else []
        self.attributes = entry.get('attributes', None)
        
        self.is_built = False
        if building:
            self.build(config)
        
        
    def inject_chunks(self, chunks: List[tuple]):
        """Inject chunks from outside, typically from on-the-fly decoding. 
        
        Notes
        -----
        In merged chunks, there may exist two chunks with same spans but different chunk-types. 
        In this case, `ck_label_ids` is crucial as input. 
        """
        assert not self.is_built
        # Do not use ```self.chunks = list(set(self.chunks + chunks))```,
        # which may return non-deterministic order?
        self.chunks = self.chunks + [ck for ck in chunks if ck not in self.chunks]
        
        
    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        """Generate negative samples from `self.chunks` and build up tensors. 
        """
        assert not self.is_built
        self.is_built = True
        
        if self.training:
            assert all(ck in self.chunks for attr_label, ck in self.attributes)
        
        # span_size_ids / ck_label_ids: (num_chunks, )
        # Note: size_id = size - 1
        self.span_size_ids = torch.tensor([end-start-1 for ck_label, start, end in self.chunks], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        self.ck_label_ids = torch.tensor([config.ck_label2idx[ck_label] for ck_label, start, end in self.chunks], dtype=torch.long)
        
        if self.attributes is not None:
            chunk2idx = {ck: k for k, ck in enumerate(self.chunks)}
            # Note: `torch.nn.BCEWithLogitsLoss` uses `float` tensor as target
            self.attr_label_ids = torch.zeros(len(self.chunks), config.attr_voc_dim, dtype=torch.float)
            for attr_label, chunk in self.attributes:
                # Note: `chunk` does not appear in `self.chunks` only if in evaluation (i.e., gold chunks missing). 
                # In this case, `attr_label_ids` is only for forward to loss, but not for backward. 
                if chunk in chunk2idx:
                    self.attr_label_ids[chunk2idx[chunk], config.attr_label2idx[attr_label]] = 1
            self.attr_label_ids[:, 0] = (self.attr_label_ids == 0).all(dim=1)
