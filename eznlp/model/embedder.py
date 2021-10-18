# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import torch

from ..token import TokenSequence
from ..nn import SinusoidPositionalEncoding
from ..nn.init import reinit_embedding_, reinit_embedding_by_pretrained_, reinit_layer_
from ..config import Config
from ..vocab import Vocab
from ..vectors import Vectors

logger = logging.getLogger(__name__)



class VocabMixin(object):
    @property
    def voc_dim(self):
        return len(self.vocab)
        
    @property
    def specials(self):
        return tuple(tok for tok, has_tok in zip(['<unk>', '<pad>', '<sos>', '<eos>'], [True, True, self.has_sos, self.has_eos]) if has_tok)
        
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
        
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
        
    @property
    def sos_idx(self):
        return self.vocab['<sos>']
        
    @property
    def eos_idx(self):
        return self.vocab['<eos>']



class OneHotConfig(Config, VocabMixin):
    """Config of an one-hot embedder.
    """
    def __init__(self, **kwargs):
        self.tokens_key = kwargs.pop('tokens_key', 'tokens')
        self.field = kwargs.pop('field')
        self.vocab = kwargs.pop('vocab', None)
        self.max_len = kwargs.pop('max_len', None)
        self.has_sos = kwargs.pop('has_sos', False)
        self.has_eos = kwargs.pop('has_eos', False)
        self.min_freq = kwargs.pop('min_freq', 1)
        
        self.emb_dim = kwargs.pop('emb_dim', 100)
        self.vectors: Vectors = kwargs.pop('vectors', None)
        if self.vectors is not None:
            if self.emb_dim != self.vectors.emb_dim:
                logger.warning(f"`emb_dim` {self.emb_dim} does not equal `vectors.emb_dim` {self.vectors.emb_dim} \n" 
                               f"Reset `emb_dim` to be {self.vectors.emb_dim}")
                self.emb_dim = self.vectors.emb_dim
        self.oov_init = kwargs.pop('oov_vector', 'zeros')
        
        self.freeze = kwargs.pop('freeze', False)
        assert not (self.freeze and self.vectors is None)
        
        self.has_positional_emb = kwargs.pop('has_positional_emb', False)
        self.sin_positional_emb = kwargs.pop('sin_positional_emb', False)
        
        self.scale_grad_by_freq = kwargs.pop('scale_grad_by_freq', False)
        super().__init__(**kwargs)
        
    @property
    def valid(self):
        return all(attr is not None for name, attr in self.__dict__.items() if name != 'vectors')
        
    @property
    def name(self):
        return self.field
        
    @property
    def out_dim(self):
        return self.emb_dim
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['vectors'] = None
        return state
        
    def _get_field(self, tokens: TokenSequence):
        x_list = getattr(tokens, self.field)
        if self.has_sos:
            x_list = ['<sos>'] + x_list
        if self.has_eos:
            x_list = x_list + ['<eos>']
        return x_list
        
    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for entry in data:
                field_seq = self._get_field(entry[self.tokens_key])
                counter.update(field_seq)
                if self.max_len is None or len(field_seq) > self.max_len:
                    self.max_len = len(field_seq)
        
        self.vocab = Vocab(counter, min_freq=self.min_freq, specials=self.specials, specials_first=True)
        
        
    def exemplify(self, tokens: TokenSequence):
        # It is generally recommended to return cpu tensors in multi-process loading. 
        # See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
        return torch.tensor([self.vocab[x] for x in self._get_field(tokens)], dtype=torch.long)
        
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
        
        if config.has_positional_emb:
            if config.sin_positional_emb:
                self.pos_embedding = SinusoidPositionalEncoding(config.max_len, config.emb_dim)
            else:
                self.pos_embedding = torch.nn.Embedding(config.max_len, config.emb_dim, scale_grad_by_freq=config.scale_grad_by_freq)
                reinit_embedding_(self.pos_embedding)
            self.register_buffer('_pos_ids', torch.arange(config.max_len))
        
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.embedding.requires_grad_(not freeze)
        
    def forward(self, x_ids: torch.LongTensor, start_position_id: int=0):
        embedded = self.embedding(x_ids)
        
        if hasattr(self, 'pos_embedding'):
            # The last dimension of `x_ids` is assumed to be step
            *batch_dims, step = x_ids.size()
            assert start_position_id + step <= self._pos_ids.size(0)
            pos_ids = self._pos_ids[start_position_id:start_position_id+step].expand(*batch_dims, -1)
            embedded = embedded + self.pos_embedding(pos_ids)
        
        return embedded




class MultiHotConfig(Config):
    """Config of a multi-hot embedder.
    """
    def __init__(self, **kwargs):
        self.field = kwargs.pop('field')
        self.in_dim = kwargs.pop('in_dim', None)
        self.use_emb = kwargs.pop('use_emb', True)
        self.emb_dim = kwargs.pop('emb_dim', 50)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self.field
        
    @property
    def out_dim(self):
        if self.use_emb:
            return self.emb_dim
        else:
            return self.in_dim
        
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
        if config.use_emb:
            # NOTE: Two reasons why this layer does not have activation. 
            # (1) Activation function should been applied after batch-norm / layer-norm. 
            # (2) This layer is semantically an embedding layer, which typically does NOT require activation. 
            self.embedding = torch.nn.Linear(config.in_dim, config.emb_dim, bias=False)
            reinit_layer_(self.embedding, 'linear')
        
    def forward(self, x_values: torch.FloatTensor):
        if hasattr(self, 'embedding'):
            return self.embedding(x_values)
        else:
            return x_values
