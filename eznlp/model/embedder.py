# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import torch
import torchtext

from ..data.token import TokenSequence
from ..nn.init import reinit_embedding_, reinit_embedding_by_pretrained_, reinit_layer_
from ..config import Config
from ..pretrained.vectors import Vectors

logger = logging.getLogger(__name__)


class OneHotConfig(Config):
    def __init__(self, **kwargs):
        self.field = kwargs.pop('field')
        self.vocab = kwargs.pop('vocab', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.vectors: Vectors = kwargs.pop('vectors', None)
        if self.vectors is not None:
            if self.emb_dim != self.vectors.emb_dim:
                logger.warning(f"`emb_dim` {self.emb_dim} does not equal `vectors.emb_dim` {self.vectors.emb_dim}" 
                               f"Reset `emb_dim` to be {self.vectors.emb_dim}")
                self.emb_dim = self.vectors.emb_dim
                
        self.oov_init = kwargs.pop('oov_vector', 'zeros')
        self.freeze = kwargs.pop('freeze', False)
        self.scale_grad_by_freq = kwargs.pop('scale_grad_by_freq', False)
        super().__init__(**kwargs)
        
    @property
    def out_dim(self):
        return self.emb_dim
        
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
    
    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                counter.update(getattr(data_entry['tokens'], self.field))
        self.vocab = torchtext.vocab.Vocab(counter, 
                                           min_freq=self.min_freq, 
                                           specials=('<unk>', '<pad>'), 
                                           specials_first=True)
        
    def exemplify(self, tokens: TokenSequence):
        # It is generally recommended to return cpu tensors in multi-process loading. 
        # See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        return torch.tensor([self.vocab[x] for x in getattr(tokens, self.field)], dtype=torch.long)
    
    def batchify(self, batch_ids: List[torch.LongTensor]):
        return torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=self.pad_idx)
        
    def instantiate(self):
        return OneHotEmbedder(self)
    
    
    
class OneHotEmbedder(torch.nn.Module):
    def __init__(self, config: OneHotConfig):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.voc_dim, 
                                            config.emb_dim, 
                                            padding_idx=config.pad_idx, 
                                            scale_grad_by_freq=config.scale_grad_by_freq)
        if config.vectors is None:
            reinit_embedding_(self.embedding)
        else:
            reinit_embedding_by_pretrained_(self.embedding, config.vocab.itos, config.vectors, config.oov_init)
            
        self.freeze = config.freeze
        self.embedding.requires_grad_(not self.freeze)
        
        
    def forward(self, x_ids: torch.LongTensor):
        return self.embedding(x_ids)
    
    
    
class MultiHotConfig(Config):
    def __init__(self, **kwargs):
        self.field = kwargs.pop('field')
        self.in_dim = kwargs.pop('in_dim', None)
        self.emb_dim = kwargs.pop('emb_dim', 50)
        super().__init__(**kwargs)
    
    @property
    def out_dim(self):
        return self.emb_dim
    
    def build_dim(self, tokens: TokenSequence):
        self.in_dim = getattr(tokens, self.field)[0].shape[0]
        
    def exemplify(self, tokens: TokenSequence):
        return torch.tensor(getattr(tokens, self.field), dtype=torch.float)
    
    def batchify(self, batch_values: List[torch.FloatTensor]):
        return torch.nn.utils.rnn.pad_sequence(batch_values, batch_first=True, padding_value=0.0)
    
    def instantiate(self):
        return MultiHotEmbedder(self)
    
    
class MultiHotEmbedder(torch.nn.Module):
    def __init__(self, config: MultiHotConfig):
        super().__init__()
        # NOTE: Two reasons why this layer does not have activation. 
        # (1) Activation function should been applied after batch-norm / layer-norm. 
        # (2) This layer is semantically an embedding layer, which typically does NOT require activation. 
        self.embedding = torch.nn.Linear(config.in_dim, config.emb_dim, bias=False)
        reinit_layer_(self.embedding, 'linear')
        
    def forward(self, x_values: torch.FloatTensor):
        return self.embedding(x_values)
    
    