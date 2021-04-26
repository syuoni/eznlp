# -*- coding: utf-8 -*-
import torch

from ..nn.init import reinit_layer_, reinit_lstm_, reinit_gru_, reinit_transformer_encoder_layer_
from ..nn.functional import mask2seq_lens
from ..nn.modules import CombinedDropout
from ..config import Config


class EncoderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'LSTM')
        self.in_dim = kwargs.pop('in_dim', None)
        self.in_proj = kwargs.pop('in_proj', False)
        self.shortcut = kwargs.pop('shortcut', False)
        
        if self.arch.lower() == 'identity':
            self.in_drop_rates = kwargs.pop('in_drop_rates', (0.0, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.0)
            
        else:
            self.hid_dim = kwargs.pop('hid_dim', 128)
            
            if self.arch.lower() in ('lstm', 'gru'):
                self.train_init_hidden = kwargs.pop('train_init_hidden', False)
                self.num_layers = kwargs.pop('num_layers', 1)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.5)
                
            elif self.arch.lower() in ('conv', 'gehring'):
                self.kernel_size = kwargs.pop('kernel_size', 3)
                self.num_layers = kwargs.pop('num_layers', 3)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.25, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.25)
                
            elif self.arch.lower() == 'transformer':
                self.nhead = kwargs.pop('nhead', 8)
                self.pf_dim = kwargs.pop('pf_dim', 256)
                self.num_layers = kwargs.pop('num_layers', 3)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.1, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.1)
                
            else:
                raise ValueError(f"Invalid encoder architecture {self.arch}")
        
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.arch
        
    @property
    def out_dim(self):
        if self.arch.lower() == 'identity':
            out_dim = self.in_dim
        else:
            out_dim = self.hid_dim
            
        if self.shortcut:
            out_dim = out_dim + self.in_dim
            
        return out_dim
        
    
    def instantiate(self):
        if self.arch.lower() == 'identity':
            return IdentityEncoder(self)
        elif self.arch.lower() in ('lstm', 'gru'):
            return RNNEncoder(self)
        elif self.arch.lower() == 'conv':
            return ConvEncoder(self)
        elif self.arch.lower() == 'gehring':
            return GehringConvEncoder(self)
        elif self.arch.lower() == 'transformer':
            return TransformerEncoder(self)
        
        
        
class Encoder(torch.nn.Module):
    """
    `Encoder` forwards from embeddings to hidden states. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.dropout = CombinedDropout(*config.in_drop_rates)
        if config.in_proj:
            self.in_proj_layer = torch.nn.Linear(config.in_dim, config.in_dim)
            reinit_layer_(self.in_proj_layer, 'linear')
        self.shortcut = config.shortcut
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
    def forward(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, 'in_proj_layer'):
            hidden = self.embedded2hidden(self.in_proj_layer(self.dropout(embedded)), mask)
        else:
            hidden = self.embedded2hidden(self.dropout(embedded), mask)
        
        if self.shortcut:
            return torch.cat([hidden, embedded], dim=-1)
        else:
            return hidden
        
        
        
class IdentityEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        return embedded
    
    
    
class RNNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        rnn_config = {'input_size': config.in_dim, 
                      'hidden_size': config.hid_dim//2, 
                      'num_layers': config.num_layers, 
                      'batch_first': True, 
                      'bidirectional': True, 
                      'dropout': 0.0 if config.num_layers <= 1 else config.hid_drop_rate}
        
        if config.arch.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config.arch.lower() == 'gru':
            self.rnn = torch.nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)
        
        # h_0/c_0: (layers*directions, batch, hid_dim/2)
        if config.train_init_hidden:
            self.h_0 = torch.nn.Parameter(torch.zeros(config.num_layers*2, 1, config.hid_dim//2))
            if isinstance(self.rnn, torch.nn.LSTM):
                self.c_0 = torch.nn.Parameter(torch.zeros(config.num_layers*2, 1, config.hid_dim//2))
                
                
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor, return_last_hidden: bool=False):
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, 
                                                                  lengths=mask2seq_lens(mask).cpu(), 
                                                                  batch_first=True, 
                                                                  enforce_sorted=False)
        if hasattr(self, 'h_0'):
            h_0 = self.h_0.repeat(1, embedded.size(0), 1)
            if hasattr(self, 'c_0'):
                c_0 = self.c_0.repeat(1, embedded.size(0), 1)
                h_0 = (h_0, c_0)
        else:
            h_0 = None
            
        if isinstance(self.rnn, torch.nn.LSTM):
            packed_rnn_outs, (h_T, _) = self.rnn(packed_embedded, h_0)
        else:
            packed_rnn_outs, h_T = self.rnn(packed_embedded, h_0)
            
        if return_last_hidden:
            # h_T: (layers*directions, batch, hid_dim/2) -> (batch, hid_dim)
            h_T = torch.cat([h_T[-2], h_T[-1]], dim=-1)
            return h_T
        else:
            # rnn_outs: (batch, step, hid_dim)
            rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_outs, batch_first=True, padding_value=0)
            return rnn_outs
        
        
    def forward2last_hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        if hasattr(self, 'in_proj_layer'):
            return self.embedded2hidden(self.in_proj_layer(self.dropout(embedded)), mask, return_last_hidden=True)
        else:
            return self.embedded2hidden(self.dropout(embedded), mask, return_last_hidden=True)
        
        
        
class ConvBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, drop_rate: float):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.relu = torch.nn.ReLU()
        reinit_layer_(self.conv, 'relu')
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        # NOTE: It would be better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layers have such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        
        # x: (batch, in_dim=channels, step)
        # mask: (batch, step)
        x.masked_fill_(mask.unsqueeze(1), 0)
        return self.relu(self.conv(self.dropout(x)))
        
        
class ConvEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # NOTE: Only the first layer is differently configured, consistent to `torch.nn.RNN` modules
        self.conv_blocks = torch.nn.ModuleList(
            [ConvBlock(in_dim=(config.in_dim if k==0 else config.hid_dim), 
                       out_dim=config.hid_dim, 
                       kernel_size=config.kernel_size, 
                       drop_rate=(0.0 if k==0 else config.hid_drop_rate)) for k in range(config.num_layers)]
        )
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        # embedded: (batch, step, emb_dim) -> (batch, emb_dim, step)
        hidden = embedded.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            hidden = conv_block(hidden, mask)
            
        # hidden: (batch, hid_dim, step) -> (batch, step, hid_dim)
        return hidden.permute(0, 2, 1)
    
    
    
class GehringConvBlock(torch.nn.Module):
    def __init__(self, hid_dim: int, kernel_size: int, drop_rate: float):
        super().__init__()
        self.conv = torch.nn.Conv1d(hid_dim, hid_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.glu = torch.nn.GLU(dim=1)
        reinit_layer_(self.conv, 'sigmoid')
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x: torch.FloatTensor, mask: torch.BoolTensor):
        # x: (batch, hid_dim=channels, step)
        # mask: (batch, step)
        x.masked_fill_(mask.unsqueeze(1), 0)
        conved = self.glu(self.conv(self.dropout(x)))
        
        # Residual connection
        return (x + conved) * (0.5**0.5)
    
    
class GehringConvEncoder(Encoder):
    """
    References
    ----------
    Gehring, J., et al. 2017. Convolutional Sequence to Sequence Learning. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim*2)
        self.glu = torch.nn.GLU(dim=-1)
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        
        self.conv_blocks = torch.nn.ModuleList(
            [GehringConvBlock(hid_dim=config.hid_dim, 
                              kernel_size=config.kernel_size, 
                              drop_rate=config.hid_drop_rate) for k in range(config.num_layers)]
        )
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        init_hidden = self.glu(self.emb2init_hid(embedded))
        
        # hidden: (batch, step, hid_dim/channels) -> (batch, hid_dim/channels, step)
        hidden = init_hidden.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            hidden = conv_block(hidden, mask)
            
        # hidden: (batch, hid_dim/channels, step) -> (batch, step, hid_dim/channels) 
        final_hidden = hidden.permute(0, 2, 1)
        # Residual connection
        return (init_hidden + final_hidden) * (0.5**0.5)
    
    
    
class TransformerEncoder(Encoder):
    """
    References
    ----------
    Vaswani, A., et al. 2017. Attention is All You Need. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim*2)
        self.glu = torch.nn.GLU(dim=-1)
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        
        self.tf_layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=config.hid_dim, 
                                                                               nhead=config.nhead, 
                                                                               dim_feedforward=config.pf_dim, 
                                                                               dropout=config.hid_drop_rate) 
                                                  for k in range(config.num_layers)])
        for tf_layer in self.tf_layers:
            reinit_transformer_encoder_layer_(tf_layer)
            
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor):
        init_hidden = self.glu(self.emb2init_hid(embedded))
        
        # hidden: (batch, step, hid_dim) -> (step, batch, hid_dim)
        hidden = init_hidden.permute(1, 0, 2)
        for tf_layer in self.tf_layers:
            hidden = tf_layer(hidden, src_key_padding_mask=mask)
        
        # hidden: (step, batch, hid_dim) -> (batch, step, hid_dim) 
        return hidden.permute(1, 0, 2)
