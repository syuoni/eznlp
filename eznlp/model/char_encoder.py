# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch
import torchtext

from ..token import TokenSequence
from ..nn.modules import SequencePooling
from ..nn.functional import seq_lens2mask
from ..nn.init import reinit_embedding_, reinit_lstm_, reinit_gru_, reinit_layer_
from ..config import Config


class CharConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CNN')
        if self.arch.lower() not in ('cnn', 'lstm', 'gru'):
            raise ValueError(f"Invalid char-level architecture {self.arch}")
            
        if self.arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            self.pooling = kwargs.pop('pooling', 'Max')
            if self.pooling.lower() not in ('max', 'mean'):
                raise ValueError(f"Invalid pooling method {self.pooling}")
                
        self.vocab = kwargs.pop('vocab', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.drop_rate = kwargs.pop('drop_rate', 0.5)
        super().__init__(**kwargs)
        
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
                for tok in data_entry['tokens'].raw_text:
                    counter.update(tok)
        self.vocab = torchtext.vocab.Vocab(counter, 
                                           min_freq=self.min_freq, 
                                           specials=('<unk>', '<pad>'), 
                                           specials_first=True)
        
    def exemplify(self, tokens: TokenSequence):
        ex_char_ids = [torch.tensor([self.vocab[x] for x in tok], dtype=torch.long) for tok in tokens.raw_text]
        return {'char_ids': ex_char_ids}
        
    def batchify(self, batch_ex: List[dict]):
        batch_char_ids = [char_ids for ex in batch_ex for char_ids in ex['char_ids']]
        tok_lens = torch.tensor([tok.size(0) for tok in batch_char_ids])
        char_mask = seq_lens2mask(tok_lens)
        
        batch_char_ids = torch.nn.utils.rnn.pad_sequence(batch_char_ids, batch_first=True, padding_value=self.pad_idx)
        return {'char_ids': batch_char_ids, 
                'tok_lens': tok_lens, 
                'char_mask': char_mask}
        
    def instantiate(self):
        if self.arch.lower() == 'cnn':
            return CharCNN(self)
        elif self.arch.lower() in ('lstm', 'gru'):
            return CharRNN(self)
        
        
class CharEncoder(torch.nn.Module):
    def __init__(self, config: CharConfig):
        super().__init__()
        self.emb = torch.nn.Embedding(config.voc_dim, config.emb_dim, padding_idx=config.pad_idx)
        self.dropout = torch.nn.Dropout(config.drop_rate)
        reinit_embedding_(self.emb)
        
        
    def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
        
    def forward(self, char_ids: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor, seq_lens: torch.Tensor):
        assert seq_lens.sum().item() == tok_lens.size(0)
        
        # embedded: (batch*tok_step, char_step, emb_dim)
        embedded = self.emb(char_ids)
        
        # hidden: (batch*tok_step, out_dim)
        hidden = self.embedded2hidden(self.dropout(embedded), tok_lens, char_mask)
        
        # Retrieve token-level steps
        # hidden: (batch, tok_step, out_dim)
        offsets = [0] + seq_lens.cumsum(dim=0).cpu().tolist()
        hidden = torch.nn.utils.rnn.pad_sequence([hidden[s:e] for s, e in zip(offsets[:-1], offsets[1:])], 
                                                 batch_first=True, 
                                                 padding_value=0.0)
        return hidden
    
    

class CharCNN(CharEncoder):
    def __init__(self, config: CharConfig):
        super().__init__(config)
        self.conv = torch.nn.Conv1d(config.emb_dim, config.out_dim, 
                              kernel_size=config.kernel_size, padding=(config.kernel_size-1)//2)
        self.relu = torch.nn.ReLU()
        self.pooling = SequencePooling(mode=config.pooling)
        reinit_layer_(self.conv, 'relu')
        
        
    def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
        # NOTE: ``embedded`` has been ensured to be zeros in padding positions.
        hidden = self.conv(embedded.permute(0, 2, 1)).permute(0, 2, 1)
        hidden = self.relu(hidden)
        
        # hidden: (batch*tok_step, char_step, out_dim) -> (batch*tok_step, out_dim)
        hidden = self.pooling(hidden, char_mask)
        return hidden
    
    
class CharRNN(CharEncoder):
    def __init__(self, config: CharConfig):
        super().__init__(config)
        
        rnn_config = {'input_size': config.emb_dim, 
                      'hidden_size': config.out_dim // 2, 
                      'batch_first': True, 
                      'bidirectional': True}
        if config.arch.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config.arch.lower() == 'gru':
            self.rnn = torch.nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)
            
            
    def embedded2hidden(self, embedded: torch.Tensor, tok_lens: torch.Tensor, char_mask: torch.Tensor):
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, tok_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        if isinstance(self.rnn, torch.nn.LSTM):
            # hidden: (num_layers*num_directions=2, batch*tok_step, hid_dim=out_dim/2)
            _, (hidden, _) = self.rnn(packed_embedded)
        else:
            _, hidden = self.rnn(packed_embedded)
        
        # hidden: (2, batch*tok_step, out_dim/2) -> (batch*tok_step, out_dim)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return hidden
    
    