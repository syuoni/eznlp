# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from .datasets_utils import Batch
from .nn_utils import reinit_embedding_, reinit_layer_, reinit_lstm_, reinit_gru_, reinit_transformer_encoder_layer_


class CharCNN(nn.Module):
    def __init__(self, voc_dim: int, emb_dim: int, out_dim: int, kernel_size: int, 
                 dropout: float, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(voc_dim, emb_dim, padding_idx=pad_idx)
        self.conv = nn.Conv1d(emb_dim, out_dim*2, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.glu = nn.GLU()
        self.dropout = nn.Dropout(dropout)
        
        reinit_embedding_(self.emb)
        reinit_layer_(self.conv, 'sigmoid')
        
        
    def forward(self, char_ids: Tensor, tok_lens: Tensor, char_mask: Tensor, seq_lens: Tensor):
        # embedded: (batch*sentence, step, emb_dim)
        embedded = self.emb(char_ids)
        
        # feats: (batch*sentence, step, out_dim)
        # NOTE: The input of this convolutional layer has been ensured to be zeros
        # in padding positions.
        feats = self.conv(self.dropout(embedded).permute(0, 2, 1)).permute(0, 2, 1)
        feats = self.glu(feats)
        
        # feats: (batch*sentence, out_dim)
        feats.masked_fill_(char_mask.unsqueeze(-1), float('-inf'))
        feats = feats.max(dim=1).values
        
        # Retrieve token-level steps
        # feats: (batch, step, out_dim)
        offsets = [0] + seq_lens.cumsum(dim=0).cpu().tolist()
        feats = pad_sequence([feats[s:e] for s, e in zip(offsets[:-1], offsets[1:])], 
                             batch_first=True, padding_value=0.0)
        return feats
    
    
class EnumEmbedding(nn.Module):
    def __init__(self, voc_dim: int, emb_dim: int, pad_idx: int):
        super().__init__()
        self.emb = nn.Embedding(voc_dim, emb_dim, padding_idx=pad_idx)
        reinit_embedding_(self.emb)
        
    def forward(self, enum_ids: Tensor):
        return self.emb(enum_ids)
    
    
class ValEmbedding(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        # NOTE: Two reasons why this layer does not have activation. 
        # (1) Activation function should been applied after batch-norm / layer-norm. 
        # (2) This layer is semantically an embedding layer, which typically does 
        # not require activation. 
        self.proj = nn.Linear(in_dim, emb_dim)
        reinit_layer_(self.proj, 'linear')
        
    def forward(self, val_ins: Tensor):
        return self.proj(val_ins)
    
    
class Encoder(nn.Module):
    def __init__(self, config: dict, itos=None, pretrained_vectors=None):
        """
        `Encoder` forwards from inputs to hidden states. 
        """
        super().__init__()
        assert config['emb_full_dim'] > 0
        self.config = config
        
        if 'tok' in config:
            self.word_emb = nn.Embedding(config['tok']['voc_dim'], config['tok']['emb_dim'], padding_idx=config['tok']['pad_idx'])
            self.pos_emb = nn.Embedding(config['max_len'], config['tok']['emb_dim'])
            reinit_embedding_(self.word_emb, itos=itos, pretrained_vectors=pretrained_vectors)
            reinit_embedding_(self.pos_emb)
            
        if 'char' in config:
            self.char_cnn = CharCNN(**config['char'])
            
        if 'enum' in config:
            self.enum_embs = nn.ModuleDict({f: EnumEmbedding(**enum_config) for f, enum_config in config['enum'].items()})
                
        if 'val' in config:
            self.val_embs = nn.ModuleDict({f: ValEmbedding(**val_config) for f, val_config in config['val'].items()})
            
        # self.embedded_layer_norm = nn.LayerNorm(config['emb_full_dim'])
        self.dropout = nn.Dropout(config['enc']['dropout'])
        
        
    def get_word_embedded(self, batch: Batch):
        if hasattr(self, 'word_emb'):
            # word_embedded: (batch, step, emb_dim)
            word_embedded = self.word_emb(batch.tok_ids)
            # pos_embedded: (batch, step, emb_dim)
            pos = torch.arange(word_embedded.size(1), device=word_embedded.device).repeat(word_embedded.size(0), 1)
            pos_embedded = self.pos_emb(pos)
            
            return [(word_embedded + pos_embedded) * (0.5**0.5)]
        else:
            return []
        
        
    def get_extra_embedded(self, batch: Batch):
        extra_embedded = []
        
        if hasattr(self, 'char_cnn'):
            extra_embedded.append(self.char_cnn(batch.char_ids, batch.tok_lens, batch.char_mask, batch.seq_lens))
        
        if hasattr(self, 'enum_embs'):
            extra_embedded.extend([self.enum_embs[f](batch.enum[f]) for f in self.enum_embs])
        
        if hasattr(self, 'val_embs'):
            extra_embedded.extend([self.val_embs[f](batch.val[f]) for f in self.val_embs])
            
        return extra_embedded
        
    
    def embedded2hidden(self, batch: Batch, full_embedded: Tensor):
        raise NotImplementedError("Not Implemented `embedded2hidden`")
        
        
    def forward(self, batch: Batch):
        # full_embedded: (batch, step, emb_dim)
        full_embedded = torch.cat(self.get_word_embedded(batch) + self.get_extra_embedded(batch), dim=-1)
        # full_embedded = self.embedded_layer_norm(full_embedded)
        
        # hidden: (batch, step, hid_dim)
        hidden = self.embedded2hidden(batch, self.dropout(full_embedded))
        if self.config['shortcut']:
            full_hidden = torch.cat([hidden, full_embedded], dim=-1)
        else:
            full_hidden = hidden
            
        return full_hidden
    
    
class RNNEncoder(Encoder):
    def __init__(self, config: dict, itos=None, pretrained_vectors=None):
        super().__init__(config, itos, pretrained_vectors)
        
        if config['enc']['arch'].lower() == 'lstm':
            self.rnn = nn.LSTM(config['emb_full_dim'], config['enc']['hid_dim']//2, num_layers=config['enc']['n_layers'], 
                               batch_first=True, dropout=config['enc']['dropout'], bidirectional=True)
            reinit_lstm_(self.rnn)
        elif config['enc']['arch'].lower() == 'gru':
            self.rnn = nn.GRU(config['emb_full_dim'], config['enc']['hid_dim']//2, num_layers=config['enc']['n_layers'], 
                              batch_first=True, dropout=config['enc']['dropout'], bidirectional=True)
            reinit_gru_(self.rnn)
        else:
            raise ValueError(f"Invalid RNN architecture: {config['arch']}")
            
        
    def embedded2hidden(self, batch: Batch, full_embedded: Tensor):
        packed_embedded = pack_padded_sequence(full_embedded, batch.seq_lens, batch_first=True, enforce_sorted=False)
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
    def __init__(self, config: dict, itos=None, pretrained_vectors=None):
        super().__init__(config, itos, pretrained_vectors)
        self.emb2init_hid = nn.Linear(config['emb_full_dim'], config['enc']['hid_dim']*2)
        self.glu = nn.GLU(dim=-1)
        self.conv_blocks = nn.ModuleList([ConvBlock(config['enc']['hid_dim'], 
                                                    config['enc']['kernel_size'], 
                                                    config['enc']['dropout']) \
                                          for _ in range(config['enc']['n_layers'])])
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        
        
    def embedded2hidden(self, batch: Batch, full_embedded: Tensor):
        init_hidden = self.glu(self.emb2init_hid(full_embedded))
        
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
    def __init__(self, config: dict, itos=None, pretrained_vectors=None):
        super().__init__(config, itos, pretrained_vectors)
        self.emb2init_hid = nn.Linear(config['emb_full_dim'], config['enc']['hid_dim']*2)
        self.glu = nn.GLU(dim=-1)
        
        self.tf_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config['enc']['hid_dim'], 
                                                                   nhead=config['enc']['nhead'], 
                                                                   dim_feedforward=config['enc']['pf_dim'], 
                                                                   dropout=config['enc']['dropout']) \
                                        for _ in range(config['enc']['n_layers'])])
        reinit_layer_(self.emb2init_hid, 'sigmoid')
        for tf_layer in self.tf_layers:
            reinit_transformer_encoder_layer_(tf_layer)
            
        
    def embedded2hidden(self, batch: Batch, full_embedded: Tensor):
        init_hidden = self.glu(self.emb2init_hid(full_embedded))
        
        # hidden: (batch, step, hid_dim) -> (step, batch, hid_dim)
        hidden = init_hidden.permute(1, 0, 2)
        for tf_layer in self.tf_layers:
            hidden = tf_layer(hidden, src_key_padding_mask=batch.tok_mask)
        
        # hidden: (step, batch, hid_dim) -> (batch, step, hid_dim) 
        return hidden.permute(1, 0, 2)
    
    
    
class BERTEncoder(nn.Module):
    def __init__(self, bert: nn.Module, config: dict):
        super().__init__()
        self.bert = bert
        self.config = config
        
    def forward(self, batch: Batch):
        # bert_outs: (batch, wp_step+2, hid_dim)
        wp_ids, wp_tok_pos = batch.wp['wp_ids'], batch.wp['wp_tok_pos']
        bert_outs, *_ = self.bert(wp_ids, attention_mask=(~batch.wp_mask).type(torch.long))
        bert_outs = bert_outs[:, 1:-1]
        
        # pos_proj: (tok_step, wp_step)
        pos_proj = torch.arange(batch.tok_ids.size(1), device=batch.tok_ids.device).unsqueeze(1).repeat(1, wp_tok_pos.size(1))
        # pos_proj: (batch, tok_step, wp_step)
        pos_proj = F.normalize((pos_proj.unsqueeze(0) == wp_tok_pos.unsqueeze(1)).type(torch.float), p=1, dim=2)
        
        # collapsed_bert_outs: (batch, tok_step, hid_dim)
        collapsed_bert_outs = pos_proj.bmm(bert_outs)
        return collapsed_bert_outs
    
    