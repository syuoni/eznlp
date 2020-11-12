# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from ..nn import MaxPooling, MeanPooling
from ..nn.init import reinit_embedding_, reinit_lstm_, reinit_gru_, reinit_layer_
from ..config import ConfigwithVocab


class CharConfig(ConfigwithVocab):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'CNN')
        
        if self.arch.lower() not in ('cnn', 'lstm', 'gru'):
            raise ValueError(f"Invalid char-level architecture {self.arch}")
            
        if self.arch.lower() == 'cnn':
            self.kernel_size = kwargs.pop('kernel_size', 3)
            self.pooling = kwargs.pop('pooling', 'Max')
            if self.pooling.lower() not in ('max', 'mean'):
                raise ValueError(f"Invalid pooling method {self.pooling}")
                
        self.emb_dim = kwargs.pop('emb_dim', 25)
        self.out_dim = kwargs.pop('out_dim', 50)
        self.drop_rate = kwargs.pop('drop_rate', 0.5)
        super().__init__(**kwargs)
    
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
        hidden = pad_sequence([hidden[s:e] for s, e in zip(offsets[:-1], offsets[1:])], batch_first=True, padding_value=0.0)
        return hidden
    
    

class CharCNN(CharEncoder):
    def __init__(self, config: CharConfig):
        super().__init__(config)
        self.conv = torch.nn.Conv1d(config.emb_dim, config.out_dim, 
                              kernel_size=config.kernel_size, padding=(config.kernel_size-1)//2)
        self.relu = torch.nn.ReLU()
        if config.pooling.lower() == 'max':
            self.pooling = MaxPooling()
        elif config.pooling.lower() == 'mean':
            self.pooling = MeanPooling()
            
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
        packed_embedded = pack_padded_sequence(embedded, tok_lens, batch_first=True, enforce_sorted=False)
        
        if isinstance(self.rnn, torch.nn.LSTM):
            # hidden: (num_layers*num_directions=2, batch*tok_step, hid_dim=out_dim/2)
            _, (hidden, _) = self.rnn(packed_embedded)
        else:
            _, hidden = self.rnn(packed_embedded)
        
        # hidden: (2, batch*tok_step, out_dim/2) -> (batch*tok_step, out_dim)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        return hidden
    
    