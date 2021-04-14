# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch
import torchtext

from ..token import TokenSequence
from ..nn.modules import SequencePooling
from ..nn.functional import seq_lens2mask
from .embedder import OneHotConfig, OneHotEmbedder
from .encoder import EncoderConfig, RNNEncoder


class NestedOneHotConfig(OneHotConfig):
    """Config for an embedder, designed for features with the structure: `step * channel * inner_step`.
    At each position exists one (or multiple) sequence. 
    
    If multiple sequences exist, they share a common vocabulary and corresponding embedding layer.
    """
    def __init__(self, **kwargs):
        self.num_channels = kwargs.pop('num_channels', 1)
        self.squeeze = kwargs.pop('squeeze', True)
        if self.squeeze:
            assert self.num_channels == 1
        
        self.encoder: EncoderConfig = kwargs.pop('encoder', None)
        self.agg_mode = kwargs.pop('agg_mode', 'mean_pooling')
        
        super().__init__(**kwargs)
        if self.encoder is not None:
            self.encoder.in_dim = self.emb_dim
        
        
    @property
    def valid(self):
        if self.encoder is not None and not self.encoder.valid:
            return False
        return all(attr is not None for name, attr in self.__dict__.items() if name not in ('vectors', 'encoder'))
        
    
    @property
    def out_dim(self):
        if self.encoder is not None:
            return self.encoder.out_dim * self.num_channels
        else:
            return self.emb_dim * self.num_channels
        
    def _inner_sequences(self, tokens: TokenSequence):
        for tok_field in getattr(tokens, self.field):
            if self.squeeze:
                yield tok_field
            else:
                yield from tok_field
                
                
    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry['tokens']):
                    counter.update(inner_seq)
        self.vocab = torchtext.vocab.Vocab(counter, 
                                           min_freq=self.min_freq, 
                                           specials=('<unk>', '<pad>'), 
                                           specials_first=True)
        
        
    def exemplify(self, tokens: TokenSequence):
        inner_ids_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_ids_list.append(torch.tensor([self.vocab[x] for x in inner_seq]))
            
        # inner_ids: (step*num_channels, inner_step)
        return {'inner_ids': inner_ids_list}
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_inner_ids = [inner_ids for ex in batch_ex for inner_ids in ex['inner_ids']]
        inner_seq_lens = torch.tensor([inner_ids.size(0) for inner_ids in batch_inner_ids])
        inner_mask = seq_lens2mask(inner_seq_lens)
        batch_inner_ids = torch.nn.utils.rnn.pad_sequence(batch_inner_ids, batch_first=True, padding_value=self.pad_idx)
        # inner_ids: (batch*step*num_channels, inner_step)
        return {'inner_ids': batch_inner_ids, 
                'inner_mask': inner_mask}
        
        
    def instantiate(self):
        return NestedOneHotEmbedder(self)
    
    
    
class NestedOneHotEmbedder(OneHotEmbedder):
    def __init__(self, config: NestedOneHotConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        if config.encoder is not None:
            self.encoder = config.encoder.instantiate()
            
        self.agg_mode = config.agg_mode
        if self.agg_mode.lower() == 'rnn_last':
            assert isinstance(self.encoder, RNNEncoder)
        elif self.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=self.agg_mode.replace('_pooling', ''))
            
        
    def _restore_outer_shapes(self, x: torch.Tensor, seq_lens: torch.LongTensor):
        offsets = [0] + (seq_lens * self.num_channels).cumsum(dim=0).cpu().tolist()
        # x: (batch, step*num_channels, emb_dim/hid_dim)
        x = torch.nn.utils.rnn.pad_sequence([x[s:e] for s, e in zip(offsets[:-1], offsets[1:])], 
                                            batch_first=True, 
                                            padding_value=0.0)
        # x: (batch, step, num_channels * emb_dim/hid_dim)
        return x.view(x.size(0), -1, self.num_channels * x.size(-1))
        
        
    def forward(self, 
                inner_ids: torch.LongTensor, 
                inner_mask: torch.BoolTensor, 
                seq_lens: torch.LongTensor, 
                inner_weight: torch.FloatTensor=None):
        assert (seq_lens * self.num_channels).sum().item() == inner_ids.size(0)
        
        # embedded: (batch*step*num_channels, inner_step, emb_dim)
        embedded = self.embedding(inner_ids)
        
        # encoding -> aggregating
        # hidden: (batch*step*num_channels, inner_step, hid_dim)
        # agg_hidden: (batch*step*num_channels, hid_dim)
        if hasattr(self, 'encoder'):
            if self.agg_mode.lower() == 'rnn_last':
                agg_hidden = self.encoder.forward2last_hidden(embedded, inner_mask)
            else:
                hidden = self.encoder(embedded, inner_mask)
                agg_hidden = self.aggregating(hidden, inner_mask, weight=inner_weight)
                
        else:
            agg_hidden = self.aggregating(embedded, inner_mask, weight=inner_weight)
            
        # Restore outer shapes (token-level steps)
        return self._restore_outer_shapes(agg_hidden, seq_lens)
        
    
    
class SoftLexiconConfig(NestedOneHotConfig):
    """
    References
    ----------
    Ma et al. (2020). Simplify the usage of lexicon in Chinese NER. ACL 2020.
    """
    def __init__(self, **kwargs):
        kwargs['field'] = kwargs.pop('field', 'softlexicon')
        kwargs['num_channels'] = kwargs.pop('num_channels', 4)
        kwargs['squeeze'] = kwargs.pop('squeeze', False)
        
        kwargs['emb_dim'] = kwargs.pop('emb_dim', 50)
        kwargs['agg_mode'] = kwargs.pop('agg_mode', 'wtd_mean_pooling')
        
        super().__init__(**kwargs)
        
        
    def build_freqs(self, *partitions):
        """
        Ma et al. (2020): The statistical data set is constructed from a combination 
        of *training* and *developing* data of the task. 
        In addition, note that the frequency of `w` does not increase if `w` is 
        covered by another sub-sequence that matches the lexicon
        """
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry['tokens']):
                    counter.update(inner_seq)
        
        # NOTE: Set the minimum frequecy as 1, to avoid OOV tokens being ignored
        self.freqs = {tok: 1 for tok in self.vocab.itos}
        self.freqs.update(counter)
        self.freqs['<pad>'] = 0
        
        
    def exemplify(self, tokens: TokenSequence):
        example = super().exemplify(tokens)
        
        inner_freqs_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_freqs_list.append(torch.tensor([self.freqs[x] for x in inner_seq]))
            
        example['inner_freqs'] = inner_freqs_list
        return example
    
    
    def batchify(self, batch_ex: List[dict]):
        batch = super().batchify(batch_ex)
        
        batch_inner_freqs = [inner_freqs for ex in batch_ex for inner_freqs in ex['inner_freqs']]
        batch_inner_freqs = torch.nn.utils.rnn.pad_sequence(batch_inner_freqs, batch_first=True, padding_value=0)
        batch['inner_weight'] = batch_inner_freqs
        return batch
        
        
        
        
class CharConfig(NestedOneHotConfig):
    def __init__(self, **kwargs):
        kwargs['field'] = kwargs.pop('field', 'raw_text')
        kwargs['num_channels'] = kwargs.pop('num_channels', 1)
        kwargs['squeeze'] = kwargs.pop('squeeze', True)
        
        kwargs['emb_dim'] = kwargs.pop('emb_dim', 16)
        kwargs['encoder'] = kwargs.pop('encoder', EncoderConfig(arch='LSTM', 
                                                                hid_dim=128, 
                                                                num_layers=1, 
                                                                in_drop_rates=(0.5, 0.0, 0.0)))
        if kwargs['encoder'].arch.lower() in ('lstm', 'gru'):
            kwargs['agg_mode'] = kwargs.pop('agg_mode', 'rnn_last')
        elif kwargs['encoder'].arch.lower() in ('conv', 'gehring'):
            kwargs['agg_mode'] = kwargs.pop('agg_mode', 'max_pooling')
            
        super().__init__(**kwargs)
        



# class CharConfig(Config):
#     def __init__(self, **kwargs):
#         self.arch = kwargs.pop('arch', 'CNN')
#         if self.arch.lower() not in ('cnn', 'lstm', 'gru'):
#             raise ValueError(f"Invalid char-level architecture {self.arch}")
            
#         if self.arch.lower() == 'cnn':
#             self.kernel_size = kwargs.pop('kernel_size', 3)
#             self.pooling = kwargs.pop('pooling', 'Max')
#             if self.pooling.lower() not in ('max', 'mean'):
#                 raise ValueError(f"Invalid pooling method {self.pooling}")
                
#         self.vocab = kwargs.pop('vocab', None)
#         self.min_freq = kwargs.pop('min_freq', 1)
#         self.emb_dim = kwargs.pop('emb_dim', 25)
#         self.out_dim = kwargs.pop('out_dim', 50)
#         self.drop_rate = kwargs.pop('drop_rate', 0.5)
#         super().__init__(**kwargs)
        
#     @property
#     def voc_dim(self):
#         return len(self.vocab)
        
#     @property
#     def pad_idx(self):
#         return self.vocab['<pad>']
        
#     @property
#     def unk_idx(self):
#         return self.vocab['<unk>']
    
#     def build_vocab(self, *partitions):
#         counter = Counter()
#         for data in partitions:
#             for data_entry in data:
#                 for tok in data_entry['tokens'].raw_text:
#                     counter.update(tok)
#         self.vocab = torchtext.vocab.Vocab(counter, 
#                                            min_freq=self.min_freq, 
#                                            specials=('<unk>', '<pad>'), 
#                                            specials_first=True)
        
#     def exemplify(self, tokens: TokenSequence):
#         ex_char_ids = [torch.tensor([self.vocab[x] for x in tok], dtype=torch.long) for tok in tokens.raw_text]
#         return {'char_ids': ex_char_ids}
        
#     def batchify(self, batch_ex: List[dict]):
#         batch_char_ids = [char_ids for ex in batch_ex for char_ids in ex['char_ids']]
#         tok_lens = torch.tensor([tok.size(0) for tok in batch_char_ids])
#         char_mask = seq_lens2mask(tok_lens)
        
#         batch_char_ids = torch.nn.utils.rnn.pad_sequence(batch_char_ids, batch_first=True, padding_value=self.pad_idx)
#         return {'char_ids': batch_char_ids, 
#                 'tok_lens': tok_lens, 
#                 'char_mask': char_mask}
        
#     def instantiate(self):
#         if self.arch.lower() == 'cnn':
#             return CharCNN(self)
#         elif self.arch.lower() in ('lstm', 'gru'):
#             return CharRNN(self)
        
        
# class CharEncoder(torch.nn.Module):
#     def __init__(self, config: CharConfig):
#         super().__init__()
#         self.emb = torch.nn.Embedding(config.voc_dim, config.emb_dim, padding_idx=config.pad_idx)
#         self.dropout = torch.nn.Dropout(config.drop_rate)
#         reinit_embedding_(self.emb)
        
        
#     def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
#         raise NotImplementedError("Not Implemented `embedded2hidden`")
        
        
#     def forward(self, char_ids: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor, seq_lens: torch.Tensor):
#         assert seq_lens.sum().item() == tok_lens.size(0)
        
#         # embedded: (batch*tok_step, char_step, emb_dim)
#         embedded = self.emb(char_ids)
        
#         # hidden: (batch*tok_step, out_dim)
#         hidden = self.embedded2hidden(self.dropout(embedded), tok_lens, char_mask)
        
#         # Retrieve token-level steps
#         # hidden: (batch, tok_step, out_dim)
#         offsets = [0] + seq_lens.cumsum(dim=0).cpu().tolist()
#         hidden = torch.nn.utils.rnn.pad_sequence([hidden[s:e] for s, e in zip(offsets[:-1], offsets[1:])], 
#                                                  batch_first=True, 
#                                                  padding_value=0.0)
#         return hidden
    
    

# class CharCNN(CharEncoder):
#     def __init__(self, config: CharConfig):
#         super().__init__(config)
#         self.conv = torch.nn.Conv1d(config.emb_dim, config.out_dim, 
#                               kernel_size=config.kernel_size, padding=(config.kernel_size-1)//2)
#         self.relu = torch.nn.ReLU()
#         self.pooling = SequencePooling(mode=config.pooling)
#         reinit_layer_(self.conv, 'relu')
        
        
#     def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
#         # NOTE: ``embedded`` has been ensured to be zeros in padding positions.
#         hidden = self.conv(embedded.permute(0, 2, 1)).permute(0, 2, 1)
#         hidden = self.relu(hidden)
        
#         # hidden: (batch*tok_step, char_step, out_dim) -> (batch*tok_step, out_dim)
#         hidden = self.pooling(hidden, char_mask)
#         return hidden
    
    
# class CharRNN(CharEncoder):
#     def __init__(self, config: CharConfig):
#         super().__init__(config)
        
#         rnn_config = {'input_size': config.emb_dim, 
#                       'hidden_size': config.out_dim // 2, 
#                       'batch_first': True, 
#                       'bidirectional': True}
#         if config.arch.lower() == 'lstm':
#             self.rnn = torch.nn.LSTM(**rnn_config)
#             reinit_lstm_(self.rnn)
#         elif config.arch.lower() == 'gru':
#             self.rnn = torch.nn.GRU(**rnn_config)
#             reinit_gru_(self.rnn)
            
            
#     def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
#         packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, tok_lens.cpu(), batch_first=True, enforce_sorted=False)
        
#         if isinstance(self.rnn, torch.nn.LSTM):
#             # hidden: (num_layers*num_directions=2, batch*tok_step, hid_dim=out_dim/2)
#             _, (hidden, _) = self.rnn(packed_embedded)
#         else:
#             _, hidden = self.rnn(packed_embedded)
        
#         # hidden: (2, batch*tok_step, out_dim/2) -> (batch*tok_step, out_dim)
#         hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
#         return hidden

