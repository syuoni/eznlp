# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .datasets_utils import Batch
from .nn_utils import reinit_layer_, reinit_lstm_, reinit_gru_, reinit_transformer_encoder_layer_


    
class Encoder(nn.Module):
    def __init__(self, config: dict):
        """
        `Encoder` forwards from embeddings to hidden states. 
        """
        super().__init__()
        assert config['emb_dim'] > 0
        self.config = config
        if 'dropout' in config:
            self.dropout = nn.Dropout(config['dropout'])
        
        
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
        
    def forward(self, batch: Batch, embedded: Tensor):
        # embedded: (batch, step, emb_dim)
        # hidden: (batch, step, hid_dim)
        if hasattr(self, 'dropout'):
            embedded = self.dropout(embedded)
        return self.embedded2hidden(batch, embedded)
    
    
    
class ShortcutEncoder(Encoder):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        # DO NOT apply dropout to shortcut
        return embedded
    
    
    
class RNNEncoder(Encoder):
    def __init__(self, config: dict):
        super().__init__(config)
        
        rnn_config = {'input_size': config['emb_dim'], 
                      'hidden_size': config['hid_dim']//2, 
                      'num_layers': config['n_layers'], 
                      'batch_first': True, 
                      'bidirectional': True, 
                      'dropout': config['dropout']}
        
        if config['arch'].lower() == 'lstm':
            self.rnn = nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config['arch'].lower() == 'gru':
            self.rnn = nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)
        else:
            raise ValueError(f"Invalid RNN architecture: {config['arch']}")
            
        
    def embedded2hidden(self, batch: Batch, embedded: Tensor):
        packed_embedded = pack_padded_sequence(embedded, batch.seq_lens, batch_first=True, enforce_sorted=False)
        packed_rnn_outs, _ = self.rnn(packed_embedded)
        # outs: (batch, step, hid_dim*num_directions)
        rnn_outs, _ = pad_packed_sequence(packed_rnn_outs, batch_first=True, padding_value=0)
        return rnn_outs
    
    
    
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
    def __init__(self, config: dict):
        super().__init__(config)
        self.emb2init_hid = nn.Linear(config['emb_dim'], config['hid_dim']*2)
        self.glu = nn.GLU(dim=-1)
        self.conv_blocks = nn.ModuleList([ConvBlock(config['hid_dim'], 
                                                    config['kernel_size'], 
                                                    config['dropout']) \
                                          for _ in range(config['n_layers'])])
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
    def __init__(self, config: dict):
        super().__init__(config)
        self.emb2init_hid = nn.Linear(config['emb_dim'], config['hid_dim']*2)
        self.glu = nn.GLU(dim=-1)
        
        self.tf_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config['hid_dim'], 
                                                                   nhead=config['nhead'], 
                                                                   dim_feedforward=config['pf_dim'], 
                                                                   dropout=config['dropout']) \
                                        for _ in range(config['n_layers'])])
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
    
    
    
class PreTrainedEncoder(nn.Module):
    def __init__(self, ptm: nn.Module):
        """
        `PreTrainedEncoder` forwards from inputs to hidden states. 
        """
        super().__init__()
        self.ptm = ptm
        
    def forward(self, batch: Batch):
        # ptm_outs: (batch, wp_step+2, hid_dim)
        wp_ids, wp_tok_pos = batch.wp['wp_ids'], batch.wp['wp_tok_pos']
        ptm_outs, *_ = self.ptm(wp_ids, attention_mask=(~batch.wp_mask).type(torch.long))
        ptm_outs = ptm_outs[:, 1:-1]
        
        # pos_proj: (tok_step, wp_step)
        pos_proj = torch.arange(batch.tok_ids.size(1), device=batch.tok_ids.device).unsqueeze(1).repeat(1, wp_tok_pos.size(1))
        # pos_proj: (batch, tok_step, wp_step)
        pos_proj = F.normalize((pos_proj.unsqueeze(0) == wp_tok_pos.unsqueeze(1)).type(torch.float), p=1, dim=2)
        
        # collapsed_ptm_outs: (batch, tok_step, hid_dim)
        collapsed_ptm_outs = pos_proj.bmm(ptm_outs)
        return collapsed_ptm_outs
    
    