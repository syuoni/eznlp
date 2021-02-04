# -*- coding: utf-8 -*-
import torch

from ..data.wrapper import Batch
from ..nn.init import reinit_layer_
from ..nn.modules import CombinedDropout
from ..config import Config


class DecoderConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        super().__init__(**kwargs)
        
    def __repr__(self):
        raise NotImplementedError("Not Implemented `__repr__`")
        
    @property
    def voc_dim(self):
        raise NotImplementedError("Not Implemented `voc_dim`")
        
    @property
    def pad_idx(self):
        raise NotImplementedError("Not Implemented `pad_idx`")
    
    
    
class Decoder(torch.nn.Module):
    def __init__(self, config: DecoderConfig):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        self.hid2logit = torch.nn.Linear(config.in_dim, config.voc_dim)
        self.dropout = CombinedDropout(*config.in_drop_rates)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `forward`")
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `decode`")
        
        