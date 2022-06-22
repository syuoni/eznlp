# -*- coding: utf-8 -*-
import torch

from ..nn.init import reinit_layer_, reinit_lstm_, reinit_gru_
from ..nn.functional import mask2seq_lens
from ..nn.modules import CombinedDropout
from ..nn.modules import FeedForwardBlock, ConvBlock, TransformerEncoderBlock
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
            
            if self.arch.lower() == 'ffn':
                self.num_layers = kwargs.pop('num_layers', 1)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.5)
                
            elif self.arch.lower() in ('lstm', 'gru'):
                self.train_init_hidden = kwargs.pop('train_init_hidden', False)
                self.num_layers = kwargs.pop('num_layers', 1)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.5)
                
            elif self.arch.lower() in ('conv', 'gehring'):
                self.kernel_size = kwargs.pop('kernel_size', 3)
                self.scale = kwargs.pop('scale', 0.5**0.5)
                self.num_layers = kwargs.pop('num_layers', 3)
                self.in_drop_rates = kwargs.pop('in_drop_rates', (0.25, 0.0, 0.0))
                self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.25)
                
            elif self.arch.lower() == 'transformer':
                self.use_emb2init_hid = kwargs.pop('use_emb2init_hid', False)
                self.num_heads = kwargs.pop('num_heads', 8)
                self.ff_dim = kwargs.pop('ff_dim', 256)
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
        elif self.arch.lower() == 'ffn':
            return FFNEncoder(self)
        elif self.arch.lower() in ('lstm', 'gru'):
            return RNNEncoder(self)
        elif self.arch.lower() == 'conv':
            return ConvEncoder(self)
        elif self.arch.lower() == 'gehring':
            return GehringConvEncoder(self)
        elif self.arch.lower() == 'transformer':
            return TransformerEncoder(self)




class Encoder(torch.nn.Module):
    """`Encoder` forwards from embeddings to hidden states. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.dropout = CombinedDropout(*config.in_drop_rates)
        if config.in_proj:
            self.in_proj_layer = torch.nn.Linear(config.in_dim, config.in_dim)
            reinit_layer_(self.in_proj_layer, 'linear')
        self.shortcut = config.shortcut
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
    def forward(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, 'in_proj_layer'):
            hidden = self.embedded2hidden(self.in_proj_layer(self.dropout(embedded)), mask=mask)
        else:
            hidden = self.embedded2hidden(self.dropout(embedded), mask=mask)
        
        if self.shortcut:
            return torch.cat([hidden, embedded], dim=-1)
        else:
            return hidden



class IdentityEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        return embedded



class FFNEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # NOTE: Only the first layer is differently configured, consistent to `torch.nn.RNN` modules
        self.ff_blocks = torch.nn.ModuleList(
            [FeedForwardBlock(in_dim=(config.in_dim if k==0 else config.hid_dim), 
                              out_dim=config.hid_dim, 
                              drop_rate=(0.0 if k==0 else config.hid_drop_rate)) for k in range(config.num_layers)]
        )
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        hidden = embedded
        for ff_block in self.ff_blocks:
            hidden = ff_block(hidden)
        
        return hidden



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
        
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None, return_last_hidden: bool=False):
        if hasattr(self, 'h_0'):
            h_0 = self.h_0.expand(-1, embedded.size(0), -1)
            if hasattr(self, 'c_0'):
                c_0 = self.c_0.expand(-1, embedded.size(0), -1)
                h_0 = (h_0, c_0)
        else:
            h_0 = None
        
        if mask is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, 
                                                               lengths=mask2seq_lens(mask).cpu(), 
                                                               batch_first=True, 
                                                               enforce_sorted=False)
        
        # rnn_outs: (batch, step, hid_dim)
        if isinstance(self.rnn, torch.nn.LSTM):
            rnn_outs, (h_T, _) = self.rnn(embedded, h_0)
        else:
            rnn_outs, h_T = self.rnn(embedded, h_0)
        
        if mask is not None:
            rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True, padding_value=0)
        
        if return_last_hidden:
            # h_T: (layers*directions, batch, hid_dim/2)
            return rnn_outs, h_T
        else:
            return rnn_outs
        
        
    def forward(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None, return_last_hidden: bool=False):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, 'in_proj_layer'):
            hidden = self.embedded2hidden(self.in_proj_layer(self.dropout(embedded)), mask=mask, return_last_hidden=return_last_hidden)
        else:
            hidden = self.embedded2hidden(self.dropout(embedded), mask=mask, return_last_hidden=return_last_hidden)
        
        if self.shortcut:
            if return_last_hidden:
                return torch.cat([hidden[0], embedded], dim=-1), hidden[1]
            else:
                return torch.cat([hidden, embedded], dim=-1)
        else:
            return hidden



class ConvEncoder(Encoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # NOTE: Only the first layer is differently configured, consistent to `torch.nn.RNN` modules
        self.conv_blocks = torch.nn.ModuleList(
            [ConvBlock(in_dim=(config.in_dim if k==0 else config.hid_dim), 
                       out_dim=config.hid_dim, 
                       kernel_size=config.kernel_size, 
                       drop_rate=(0.0 if k==0 else config.hid_drop_rate), 
                       nonlinearity='relu') for k in range(config.num_layers)]
        )
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        # embedded: (batch, step, emb_dim) -> (batch, emb_dim, step)
        hidden = embedded.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            hidden = conv_block(hidden, mask=mask)
        
        # hidden: (batch, hid_dim, step) -> (batch, step, hid_dim)
        return hidden.permute(0, 2, 1)



class GehringConvEncoder(Encoder):
    """Convolutional sequence encoder by Gehring et al. (2017). 
    
    References
    ----------
    Gehring, J., et al. 2017. Convolutional Sequence to Sequence Learning. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim*2)
        self.glu = torch.nn.GLU(dim=-1)
        reinit_layer_(self.emb2init_hid, 'glu')
        
        self.conv_blocks = torch.nn.ModuleList(
            [ConvBlock(in_dim=config.hid_dim, 
                       out_dim=config.hid_dim, 
                       kernel_size=config.kernel_size, 
                       drop_rate=config.hid_drop_rate, # Note to apply dropout to `init_hidden`
                       nonlinearity='glu') for k in range(config.num_layers)]
        )
        self.scale = config.scale
        
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        init_hidden = self.glu(self.emb2init_hid(embedded))
        
        # hidden: (batch, step, hid_dim/channels) -> (batch, hid_dim/channels, step)
        hidden = init_hidden.permute(0, 2, 1)
        for conv_block in self.conv_blocks:
            conved = conv_block(hidden, mask=mask)
            hidden = (hidden + conved) * self.scale
        
        # hidden: (batch, hid_dim/channels, step) -> (batch, step, hid_dim/channels) 
        final_hidden = hidden.permute(0, 2, 1)
        # Residual connection
        return (init_hidden + final_hidden) * self.scale



# TODO: Initialization with (truncated) normal distribution with standard deviation of 0.02?
class TransformerEncoder(Encoder):
    """Transformer encoder by Vaswani et al. (2017). 
    
    References
    ----------
    Vaswani, A., et al. 2017. Attention is All You Need. 
    """
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        if config.use_emb2init_hid:
            self.emb2init_hid = torch.nn.Linear(config.in_dim, config.hid_dim)
            self.relu = torch.nn.ReLU()
            reinit_layer_(self.emb2init_hid, 'relu')
        else:
            assert config.hid_dim == config.in_dim
        
        self.tf_blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(hid_dim=config.hid_dim, 
                                     ff_dim=config.ff_dim, 
                                     num_heads=config.num_heads, 
                                     drop_rate=(0.0 if (k==0 and not config.use_emb2init_hid) else config.hid_drop_rate), 
                                     nonlinearity='relu') for k in range(config.num_layers)]
        )
        
    def embedded2hidden(self, embedded: torch.FloatTensor, mask: torch.BoolTensor=None):
        if hasattr(self, 'emb2init_hid'):
            hidden = self.relu(self.emb2init_hid(embedded))
        else:
            hidden = embedded
        
        for tf_block in self.tf_blocks:
            hidden = tf_block(hidden, mask=mask)
        
        return hidden
