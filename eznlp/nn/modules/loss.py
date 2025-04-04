# -*- coding: utf-8 -*-
import torch

from ..functional import (
    focal_loss,
    smooth_label_cross_entropy,
    soft_label_cross_entropy,
)


class SoftLabelCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight: torch.Tensor = None, reduction: str = "none"):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)

    def extra_repr(self):
        return f"weight={self.weight}"

    def forward(self, logits: torch.Tensor, soft_target: torch.Tensor):
        return soft_label_cross_entropy(
            logits, soft_target, weight=self.weight, reduction=self.reduction
        )


class SmoothLabelCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        epsilon: float = 0.1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "none",
    ):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def extra_repr(self):
        return f"epsilon={self.epsilon}, weight={self.weight}"

    def forward(self, logits: torch.Tensor, target: torch.LongTensor):
        return smooth_label_cross_entropy(
            logits,
            target,
            epsilon=self.epsilon,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        gamma: float = 0.0,
        weight: float = None,
        ignore_index: int = -100,
        reduction: str = "none",
    ):
        weight = (
            weight
            if weight is None or isinstance(weight, torch.Tensor)
            else torch.tensor(weight)
        )
        super().__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def extra_repr(self):
        return f"gamma={self.gamma}, weight={self.weight}"

    def forward(self, logits: torch.Tensor, target: torch.LongTensor):
        return focal_loss(
            logits,
            target,
            gamma=self.gamma,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
