# -*- coding: utf-8 -*-
from typing import List, Union
import torch

from ...wrapper import Batch
from ...config import Config
from ...nn.modules import SmoothLabelCrossEntropyLoss, FocalLoss


class DecoderMixinBase(object):
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
        """Calculate the metric (i.e., accuracy or F1) evaluating the predicted results against the gold results. 
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



class SingleDecoderConfigBase(Config):
    def __init__(self, **kwargs):
        self.in_dim = kwargs.pop('in_dim', None)
        
        # whether to allow multi-label prediction
        self.multilabel = kwargs.pop('multilabel', False)
        self.conf_thresh = kwargs.pop('conf_thresh', 0.5)
        # focal loss `gamma`: 0 fallback to cross entropy
        self.fl_gamma = kwargs.pop('fl_gamma', 0.0) 
        # label smoothing `epsilon`: 0 fallback to cross entropy
        self.sl_epsilon = kwargs.pop('sl_epsilon', 0.0)
        super().__init__(**kwargs)
        
    @property
    def criterion(self):
        if self.fl_gamma > 0:
            crit_name = f"FL({self.fl_gamma:.2f})"
        elif self.sl_epsilon > 0:
            crit_name = f"SL({self.sl_epsilon:.2f})"
        else:
            crit_name = "CE"
        return f"B{crit_name}" if self.multilabel else crit_name
        
    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith('b'):
            if self.criterion.lower().startswith('bce'):
                return torch.nn.BCEWithLogitsLoss(**kwargs)
            else:
                raise ValueError(f"Not implemented criterion: {self.criterion}")
        else:
            if self.criterion.lower().startswith('fl'):
                return FocalLoss(gamma=self.fl_gamma, **kwargs)
            elif self.criterion.lower().startswith('sl'):
                return SmoothLabelCrossEntropyLoss(epsilon=self.sl_epsilon, **kwargs)
            else:
                return torch.nn.CrossEntropyLoss(**kwargs)



class DecoderBase(torch.nn.Module):
    def __init__(self):
        """`Decoder` forwards from hidden states to outputs. 
        """
        super().__init__()
        
    def forward(self, batch: Batch, **states):
        raise NotImplementedError("Not Implemented `forward`")
        
    def decode(self, batch: Batch, **states):
        raise NotImplementedError("Not Implemented `decode`")
        
    # TODO: Loosely decoding
    def loosely_decode(self, batch: Batch, **states):
        return self.decode(batch, **states)
        
    def _unsqueezed_decode(self, batch: Batch, **states):
        if self.num_metrics == 0:
            raise RuntimeError("`_unsqueezed` method does not applies if `num_metrics` is 0")
        elif self.num_metrics == 1:
            return (self.decode(batch, **states), )
        else:
            return self.decode(batch, **states)
