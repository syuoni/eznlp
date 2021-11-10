# -*- coding: utf-8 -*-
import torch

from ..init import reinit_layer_
from ..utils import _nonlinearity2activation
from .attention import MultiheadAttention


class FeedForwardBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, drop_rate: float=0.5, nonlinearity: str='relu'):
        super().__init__()
        self.proj_layer = torch.nn.Linear(in_dim, out_dim)
        self.activation = _nonlinearity2activation(nonlinearity)
        reinit_layer_(self.proj_layer, nonlinearity)
        self.dropout = torch.nn.Dropout(drop_rate)
        
    def forward(self, x: torch.Tensor):
        return self.activation(self.proj_layer(self.dropout(x)))



class ConvBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, padding_mode: str='both', drop_rate: float=0.5, nonlinearity: str='relu'):
        super().__init__()
        padding_size = 0 if padding_mode.lower() == 'none' else kernel_size-1
        
        if nonlinearity.lower() == 'glu':
            self.conv = torch.nn.Conv1d(in_dim, out_dim*2, kernel_size=kernel_size, padding=padding_size)
            self.activation = torch.nn.GLU(dim=1)
        else:
            self.conv = torch.nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding_size)
            self.activation = _nonlinearity2activation(nonlinearity)
        
        reinit_layer_(self.conv, nonlinearity)
        self.dropout = torch.nn.Dropout(drop_rate)
        
        assert padding_mode.lower() in ('both', 'pre', 'post', 'none')
        self.padding_mode = padding_mode
        
    @property
    def kernel_size(self):
        return self.conv.kernel_size[0]
        
    @property
    def _pre_trim_size(self):
        if self.padding_mode.lower() == 'both':
            return (self.kernel_size - 1) // 2
        elif self.padding_mode.lower() == 'pre':
            # If paddings are in front, do not trim the front tensors; vice versa. 
            return 0
        else:
            return self.kernel_size - 1
        
    @property
    def _post_trim_size(self):
        return self.kernel_size - 1 - self._pre_trim_size
        
    def _trim(self, x: torch.Tensor):
        assert self.padding_mode.lower() in ('both', 'pre', 'post'), f"Illegal to trim with `padding_mode` {self.padding_mode}"
        assert x.dim() == 3
        # x: (batch, channels, step)
        return x[:, :, self._pre_trim_size:x.size(-1)-self._post_trim_size]
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        # NOTE: It would be better to ensure the input (rather than the output) of convolutional
        # layers to be zeros in padding positions, since only convolutional layers have such 
        # a property: Its output values in non-padding positions are sensitive to the input
        # values in padding positions. 
        
        # x: (batch, in_dim=channels, step)
        # mask: (batch, step)
        if mask is not None:
            x.masked_fill_(mask.unsqueeze(1), 0)
        
        # conved: (batch, out_dim=channels, step)
        conved = self.activation(self.conv(self.dropout(x)))
        if self.padding_mode.lower() == 'none':
            return conved
        else:
            return self._trim(conved)



class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, hid_dim: int, ff_dim: int, num_heads: int=8, scoring: str='scaled_dot', drop_rate: float=0.1, nonlinearity: str='relu'):
        super().__init__()
        self.self_attention = MultiheadAttention(hid_dim, num_heads=num_heads, scoring=scoring, drop_rate=drop_rate)
        self.self_norm = torch.nn.LayerNorm(hid_dim)
        
        self.ff1 = torch.nn.Linear(hid_dim, ff_dim)
        # reinit_layer_(self.ff1, nonlinearity)
        reinit_layer_(self.ff1, 'linear')
        self.activation = _nonlinearity2activation(nonlinearity)
        self.ff2 = torch.nn.Linear(ff_dim, hid_dim)
        reinit_layer_(self.ff2, 'linear')
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        
        self.dropout = torch.nn.Dropout(drop_rate)
        
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor=None, return_atten_weight: bool=False):
        attened, atten_weight = self.self_attention(self.dropout(x), self.dropout(x), self.dropout(x), mask=mask, return_atten_weight=True)
        attened_x = self.self_norm(self.dropout(x) + self.dropout(attened))
        
        ffed = self.ff2(self.dropout(self.activation(self.ff1(attened_x))))
        ffed_attened_x = self.ff_norm(attened_x + self.dropout(ffed))
        if return_atten_weight:
            return ffed_attened_x, atten_weight
        else:
            return ffed_attened_x



class TransformerDecoderBlock(torch.nn.Module):
    def __init__(self, hid_dim: int, ff_dim: int, ctx_dim: int=None, num_heads: int=8, scoring: str='scaled_dot', drop_rate: float=0.1, nonlinearity: str='relu'):
        super().__init__()
        self.self_attention = MultiheadAttention(hid_dim, num_heads=num_heads, scoring=scoring, drop_rate=drop_rate)
        self.self_norm = torch.nn.LayerNorm(hid_dim)
        
        self.cross_attention = MultiheadAttention(hid_dim, key_dim=ctx_dim, value_dim=ctx_dim, num_heads=num_heads, scoring=scoring, drop_rate=drop_rate)
        self.cross_norm = torch.nn.LayerNorm(hid_dim)
        
        self.ff1 = torch.nn.Linear(hid_dim, ff_dim)
        # reinit_layer_(self.ff1, nonlinearity)
        reinit_layer_(self.ff1, 'linear')
        self.activation = _nonlinearity2activation(nonlinearity)
        self.ff2 = torch.nn.Linear(ff_dim, hid_dim)
        reinit_layer_(self.ff2, 'linear')
        self.ff_norm = torch.nn.LayerNorm(hid_dim)
        
        self.dropout = torch.nn.Dropout(drop_rate)
        
        # `trg_mask` masks subsequent/future tokens, which is a matrix where
        # rows represent queries, and columns represent keys/values. 
        # Note the last query token can observe all tokens (last row is all False) 
        # | F T T T ... T |
        # | F F T T ... T |
        # | F F F T ... T |
        # | ... ... ... . |
        # | F F F F ... F |
        self.register_buffer('_trg_mask', torch.ones(100, 100, dtype=torch.bool).triu(diagonal=1))
        
        
    def _get_trg_mask(self, seq_len: int):
        if self._trg_mask.size(0) < seq_len:
            self.register_buffer('_trg_mask', torch.ones(seq_len*2, seq_len*2, dtype=torch.bool, device=self._trg_mask.device).triu(diagonal=1))
        return self._trg_mask[:seq_len, :seq_len]
        
    def forward(self, x: torch.Tensor, src_x: torch.Tensor, src_mask: torch.Tensor=None, last_step: bool=False, return_atten_weight: bool=False):
        # x: (batch, trg_step, hid_dim)
        #     Targets as queries/keys/values in self-attention. 
        # src_x: (batch, src_step, hid_dim)
        #     Sources as keys/values in cross-attention. 
        # src_mask: (batch, src_step)
        
        if last_step:
            # Use the last step of `x` only as the query
            # xq: (batch, trg_step=1, hid_dim)
            xq = x[:, -1:]
            trg_mask = None
        else:
            xq = x
            # trg_mask: (batch, trg_step, trg_step)
            trg_mask = self._get_trg_mask(x.size(1)).expand(x.size(0), -1, -1)
        
        attened, atten_weight = self.self_attention(self.dropout(xq), self.dropout(x), self.dropout(x), mask=trg_mask, return_atten_weight=True)
        attened_xq = self.self_norm(self.dropout(xq) + self.dropout(attened))
        
        crossed, cross_atten_weight = self.cross_attention(attened_xq, self.dropout(src_x), self.dropout(src_x), mask=src_mask, return_atten_weight=True)
        crossed_attened_xq = self.cross_norm(attened_xq + self.dropout(crossed))
        
        ffed = self.ff2(self.dropout(self.activation(self.ff1(crossed_attened_xq))))
        ffed_crossed_attened_xq = self.ff_norm(crossed_attened_xq + self.dropout(ffed))
        if return_atten_weight:
            return ffed_crossed_attened_xq, atten_weight, cross_atten_weight
        else:
            return ffed_crossed_attened_xq
