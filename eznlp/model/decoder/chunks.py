# -*- coding: utf-8 -*-
from typing import List, Union
import random
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
        
        self.num_tokens = len(entry['tokens'])
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
            # Do not use ```self.chunks = list(set(self.chunks_gold + self.chunks_pred))```,
            # which may return non-deterministic order. 
            # In the early training stage, the chunk-decoder may produce too many predicted chunks, so do sampling here. 
            chunks_extra = [ck for ck in self.chunks_pred if ck not in self.chunks_gold]
            num_neg_chunks = max(len(self.chunks_gold), int(self.num_tokens*0.2)-len(self.chunks_gold), 10)
            if len(chunks_extra) > num_neg_chunks:
                chunks_extra = random.sample(chunks_extra, num_neg_chunks)
            self.chunks = self.chunks_gold + chunks_extra
            
            # In merged chunks, there may exist two chunks with a same span but different chunk-types. 
            # In this case, auxiliary chunk-type embeddings may be useful. 
            self.chunk2idx = {ck: i for i, ck in enumerate(self.chunks)}
        
        
    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_chunks = len(self.chunks)
        
        self.span_size_ids = torch.tensor([end-start-1 for label, start, end in self.chunks], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids > config.max_size_id, config.max_size_id)
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
                # Here `head`/`tail` may not appear in `chunks`, only in the evaluation phase where `chunks_gold` are not allowed to access. 
                # In this case, `cp2label_id` is only for forwarding to a "fake" loss, but not for backwarding. 
                if (not self.training) and (head not in self.chunk2idx or tail not in self.chunk2idx):
                    continue
                hk = self.chunk2idx[head]
                tk = self.chunk2idx[tail]
                self.cp2label_id[hk, tk] = config.label2idx[label]



class ChunkSingles(TargetWrapper):
    """A wrapper of chunk-singles with underlying attributes. 
    This object enumerates all positive spans (entity spans). 
    
    For pipeline modeling, `chunks_pred` is pre-computed, and thus initially non-empty in `entry`;
    For joint modeling, `chunks_pred` is computed on-the-fly, and thus initially empty in `entry`.
    
    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple], 
         'attributes': List[tuple]}
    """
    def __init__(self, entry: dict, config: SingleDecoderConfigBase, training: bool=True):
        super().__init__(training)
        
        self.num_tokens = len(entry['tokens'])
        self.chunks_gold = entry['chunks'] if training else []
        self.chunks_pred = entry.get('chunks_pred', None)
        self.attributes = entry.get('attributes', None)
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
            # Do not use ```self.chunks = list(set(self.chunks_gold + self.chunks_pred))```,
            # which may return non-deterministic order. 
            # In the early training stage, the chunk-decoder may produce too many predicted chunks, so do sampling here. 
            chunks_extra = [ck for ck in self.chunks_pred if ck not in self.chunks_gold]
            num_neg_chunks = max(len(self.chunks_gold), int(self.num_tokens*0.2)-len(self.chunks_gold), 10)
            if len(chunks_extra) > num_neg_chunks:
                chunks_extra = random.sample(chunks_extra, num_neg_chunks)
            self.chunks = self.chunks_gold + chunks_extra
            
            # In merged chunks, there may exist two chunks with a same span but different chunk-types. 
            # In this case, auxiliary chunk-type embeddings may be useful. 
            self.chunk2idx = {ck: i for i, ck in enumerate(self.chunks)}
        
        
    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_chunks = len(self.chunks)
        
        self.span_size_ids = torch.tensor([end-start-1 for label, start, end in self.chunks], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids > config.max_size_id, config.max_size_id)
        self.ck_label_ids = torch.tensor([config.ck_label2idx[label] for label, start, end in self.chunks], dtype=torch.long)
        
        if self.training and config.neg_sampling_rate < 1:
            non_mask_rate = config.neg_sampling_rate * torch.ones(num_chunks, dtype=torch.float)
            for label, chunk in self.attributes:
                k = self.chunk2idx[chunk]
                non_mask_rate[k] = 1
            
            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool() 
        
        if self.attributes is not None:
            # `torch.nn.BCEWithLogitsLoss` uses float tensor as target
            self.cs2label_id = torch.zeros(num_chunks, config.voc_dim, dtype=torch.float)
            for label, chunk in self.attributes:
                # Here `chunk` may not appear in `chunks`, only in the evaluation phase where `chunks_gold` are not allowed to access. 
                # In this case, `cs2label_id` is only for forwarding to a "fake" loss, but not for backwarding. 
                if (not self.training) and (chunk not in self.chunk2idx):
                    continue
                k = self.chunk2idx[chunk]
                self.cs2label_id[k, config.label2idx[label]] = 1
            
            # Assign `<none>` label
            self.cs2label_id[:, config.none_idx] = (self.cs2label_id == 0).all(dim=1)
