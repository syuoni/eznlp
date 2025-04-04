# -*- coding: utf-8 -*-
from typing import List, Union

import torch

from ..functional import (
    rnn_last_selecting,
    sequence_group_aggregating,
    sequence_pooling,
)


class SequencePooling(torch.nn.Module):
    """Pooling values over steps.

    Parameters
    ----------
    x: torch.FloatTensor (batch, step, hid_dim)
    mask: torch.BoolTensor (batch, step)
    mode: str
        'mean', 'max', 'min', 'wtd_mean', 'rnn_last'
    """

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode.lower() not in ("mean", "max", "min", "wtd_mean", "rnn_last"):
            raise ValueError(f"Invalid pooling mode {mode}")
        self.mode = mode

    def forward(
        self,
        x: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        weight: torch.FloatTensor = None,
    ):
        if self.mode.lower() == "rnn_last":
            return rnn_last_selecting(x, mask)
        else:
            return sequence_pooling(x, mask, weight=weight, mode=self.mode)

    def extra_repr(self):
        return f"mode={self.mode}"


class SequenceGroupAggregating(torch.nn.Module):
    """Aggregating values over steps by groups.

    Parameters
    ----------
    x : torch.FloatTensor (batch, ori_step, hidden)
        The tensor to be aggregate.
    group_by : torch.LongTensor (batch, ori_step)
        The tensor indicating the positions after aggregation.
        Positions being negative values are NOT used in aggregation.
    agg_mode: str
        'mean', 'max', 'min', 'first', 'last'
    agg_step: int
    """

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode.lower() not in ("mean", "max", "min", "first", "last"):
            raise ValueError(f"Invalid aggregating mode {mode}")
        self.mode = mode

    def forward(
        self, x: torch.FloatTensor, group_by: torch.LongTensor, agg_step: int = None
    ):
        return sequence_group_aggregating(
            x, group_by, agg_mode=self.mode, agg_step=agg_step
        )

    def extra_repr(self):
        return f"mode={self.mode}"


class ScalarMix(torch.nn.Module):
    """Mix multi-layer hidden states by corresponding scalar weights.

    Computes a parameterised scalar mixture of N tensors,
    ``mixture = gamma * \sum_k(s_k * tensor_k)``
    where ``s = softmax(w)``, with `w` and `gamma` scalar parameters.

    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations.
    [2] https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
    """

    def __init__(self, mix_dim: int):
        super().__init__()
        self.scalars = torch.nn.Parameter(torch.zeros(mix_dim))

    def __repr__(self):
        return f"{self.__class__.__name__}(mix_dim={self.scalars.size(0)})"

    def forward(self, tensors: Union[torch.FloatTensor, List[torch.FloatTensor]]):
        if isinstance(tensors, (list, tuple)):
            tensors = torch.stack(tensors)

        norm_weights_shape = tuple([-1] + [1] * (tensors.dim() - 1))
        norm_weights = torch.nn.functional.softmax(self.scalars, dim=0).view(
            *norm_weights_shape
        )
        return (tensors * norm_weights).sum(dim=0)
