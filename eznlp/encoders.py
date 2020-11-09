# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .datasets_utils import Batch
from .nn_utils import reinit_layer_, reinit_lstm_, reinit_gru_, reinit_transformer_encoder_layer_
from .nn import WordDropout, LockedDropout
from .config import Config


class EncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'LSTM')
        self.in_dim = kwargs.pop('in_dim', None)
        self.word_dropout = kwargs.pop('word_dropout', 0.05)
        self.locked_dropout = kwargs.pop('locked_dropout', 0.5)
        
        if self.arch.lower() == 'shortcut':
            self.hid_dim = kwargs.pop('hid_dim', None)
            # DO NOT apply dropout to shortcut
            self.dropout = kwargs.pop('dropout', 0.0)
            
        elif self.arch.lower() in ('lstm', 'gru'):
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.trainable_initial_hidden = kwargs.pop('trainable_initial_hidden', True)
            self.num_layers = kwargs.pop('num_layers', 1)
            self.dropout = kwargs.pop('dropout', 0.5)
            
        elif self.arch.lower() == 'cnn':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.kernel_size = kwargs.pop('kernel_size', 3)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.25)
            
        elif self.arch.lower() == 'transformer':
            self.hid_dim = kwargs.pop('hid_dim', 128)
            self.nhead = kwargs.pop('nhead', 8)
            self.pf_dim = kwargs.pop('pf_dim', 256)
            self.num_layers = kwargs.pop('num_layers', 3)
            self.dropout = kwargs.pop('dropout', 0.1)
            
        else:
            raise ValueError(f"Invalid encoder architecture {self.arch}")
        
        super().__init__(**kwargs)
        
        
    def instantiate(self):
        if self.arch.lower() == 'shortcut':
            return ShortcutEncoder(self)
        elif self.arch.lower() in ('lstm', 'gru'):
            return RNNEncoder(self)
        elif self.arch.lower() == 'cnn':
            return CNNEncoder(self)
        elif self.arch.lower() == 'transformer':
            return TransformerEncoder(self)
        
        
        
class Encoder(nn.Module):
    """
    `Encoder` forwards from embeddings to hidden states. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        # TODO: Only applies to embeddings?
        if config.word_dropout > 0:
            self.word_dropout = WordDropout(config.word_dropout)
        if config.locked_dropout > 0:
            self.locked_dropout = LockedDropout(config.locked_dropout)
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
            
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
        
    def forward(self, batch: Batch, embedded: Tensor):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, 'dropout'):
            embedded = self.dropout(embedded)
        if hasattr(self, 'word_dropout'):
            embedded = self.word_dropout(embedded)
        if hasattr(self, 'locked_dropout'):
            embedded = self.locked_dropout(embedded)
            
        return self.embedded2hidden(batch, embedded)
    
    
    
class ShortcutEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
    
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        return embedded
    
    
# TODO: Isolate RNN with hidden0, to reuse in other modules
class RNNEncoder(Encoder):
    def __init__(self, config: dict):
        super().__init__(config)
        
        rnn_config = {'input_size': config.in_dim, 
                      'hidden_size': config.hid_dim//2, 
                      'num_layers': config.num_layers, 
                      'batch_first': True, 
                      'bidirectional': True, 
                      'dropout': 0.0 if config.num_layers <= 1 else config.dropout}
        
        if config.arch.lower() == 'lstm':
            self.rnn = nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
            
        elif config.arch.lower() == 'gru':
            self.rnn = nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)
            
        if config.trainable_initial_hidden:
            # h_0/c_0: (num_layers * num_directions, batch, hidden_size)
            self.h_0 = nn.Parameter(torch.zeros(config.num_layers*2, 1, config.hid_dim//2))
        if config.trainable_initial_hidden and config.arch.lower() == 'lstm':
            self.c_0 = nn.Parameter(torch.zeros(config.num_layers*2, 1, config.hid_dim//2))
            
        
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        packed_embedded = pack_padded_sequence(embedded, batch.seq_lens, batch_first=True, enforce_sorted=False)
        
        if hasattr(self, 'h_0'):
            h_0 = self.h_0.repeat(1, batch.tok_ids.size(0), 1)
            if hasattr(self, 'c_0'):
                c_0 = self.c_0.repeat(1, batch.tok_ids.size(0), 1)
                h_0 = (h_0, c_0)
            packed_rnn_outs, _ = self.rnn(packed_embedded, h_0)
        else:
            packed_rnn_outs, _ = self.rnn(packed_embedded)
        # outs: (batch, step, hid_dim*num_directions)
        rnn_outs, _ = pad_packed_sequence(packed_rnn_outs, batch_first=True, padding_value=0)
        return rnn_outs
    

# TODO: More CNN structures? to reuse in other modules
class ConvBlock(nn.Module):
    def __init__(self, hid_dim: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(hid_dim, hid_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.glu = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)
        reinit_layer_(self.conv, 'sigmoid')
        
        
    def forward(self, hidden: Tensor, mask: Tensor):
        # hidden: (batch, hid_dim=channels, step)
        # mask: (batch, step)
        # NOTE: It is better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layer has such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        hidden.masked_fill_(mask.unsqueeze(1), 0)
        conved = self.glu(self.conv(self.dropout(hidden)))
        
        # Residual connection
        return (hidden + conved) * (0.5**0.5)
    
    
class CNNEncoder(Encoder):
    """
    Gehring, J., et al. 2017. Convolutional Sequence to Sequence Learning. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = nn.Linear(config.in_dim, config.hid_dim*2)
        self.glu = nn.GLU(dim=-1)
        self.conv_blocks = nn.ModuleList([ConvBlock(config.hid_dim, config.kernel_size, config.dropout) \
                                          for _ in range(config.num_layers)])
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        
        
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        init_hidden = self.glu(self.emb2init_hid(embedded))
        
        # hidden: (batch, step, hid_dim/channels) -> (batch, hid_dim/channels, step)
        hidden = init_hidden.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            hidden = conv_block(hidden, batch.tok_mask)
            
        # hidden: (batch, hid_dim/channels, step) -> (batch, step, hid_dim/channels) 
        final_hidden = hidden.permute(0, 2, 1)
        # Residual connection
        return (init_hidden + final_hidden) * (0.5**0.5)
    
    
class TransformerEncoder(Encoder):
    """
    Vaswani, A., et al. 2017. Attention is All You Need. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = nn.Linear(config.in_dim, config.hid_dim*2)
        self.glu = nn.GLU(dim=-1)
        
        self.tf_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.hid_dim, 
                                                                   nhead=config.nhead, 
                                                                   dim_feedforward=config.pf_dim, 
                                                                   dropout=config.dropout) \
                                        for _ in range(config.num_layers)])
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        for tf_layer in self.tf_layers:
            reinit_transformer_encoder_layer_(tf_layer)
            
        
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        init_hidden = self.glu(self.emb2init_hid(embedded))
        
        # hidden: (batch, step, hid_dim) -> (step, batch, hid_dim)
        hidden = init_hidden.permute(1, 0, 2)
        for tf_layer in self.tf_layers:
            hidden = tf_layer(hidden, src_key_padding_mask=batch.tok_mask)
        
        # hidden: (step, batch, hid_dim) -> (batch, step, hid_dim) 
        return hidden.permute(1, 0, 2)
    
    
    