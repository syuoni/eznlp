# -*- coding: utf-8 -*-
from typing import List, Union
import torch

from ...wrapper import Batch
from ...config import Config


class DecoderMixin(object):
    @property
    def num_metrics(self):
        return 1
        
    def retrieve(self, batch: Batch):
        raise NotImplementedError("Not Implemented `retrieve`")
        
    def _unsqueezed_retrieve(self, batch: Batch):
        if self.num_metrics == 0:
            raise RuntimeError("`_unsqueezed` method does not applies if `num_metrics` is 0")
        elif self.num_metrics == 1:
            return (self.retrieve(batch), )
        else:
            return self.retrieve(batch)
        
    def evaluate(self, y_gold: Union[list, List[list]], y_pred: Union[list, List[list]]):
        """
        Calculate the metric (i.e., accuracy or F1) evaluating the predicted results against the gold results. 
        This method should typically evaluate over a full dataset, although it also compatibly evaluates over a batch. 
        """
        raise NotImplementedError("Not Implemented `evaluate`")
        
    def _unsqueezed_evaluate(self, y_gold: List[list], y_pred: List[list]):
        if self.num_metrics == 0:
            raise RuntimeError("`_unsqueezed` method does not applies if `num_metrics` is 0")
        else:
            assert len(y_gold) == self.num_metrics
            assert len(y_pred) == self.num_metrics
            
        if self.num_metrics == 1:
            return (self.evaluate(y_gold[0], y_pred[0]), )
        else:
            return self.evaluate(y_gold, y_pred)



class DecoderConfig(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        super().__init__(**kwargs)



class Decoder(torch.nn.Module):
    def __init__(self):
        """
        `Decoder` forward from hidden states to outputs. 
        """
        super().__init__()
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `forward`")
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        raise NotImplementedError("Not Implemented `decode`")
        
    def _unsqueezed_decode(self, batch: Batch, full_hidden: torch.Tensor):
        if self.num_metrics == 0:
            raise RuntimeError("`_unsqueezed` method does not applies if `num_metrics` is 0")
        elif self.num_metrics == 1:
            return (self.decode(batch, full_hidden), )
        else:
            return self.decode(batch, full_hidden)
