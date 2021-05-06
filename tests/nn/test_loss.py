# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn import SmoothLabelCrossEntropyLoss, FocalLoss


@pytest.mark.parametrize("weight", [None, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])])
@pytest.mark.parametrize("ignore_index", [-100, 0])
@pytest.mark.parametrize("reduction", ['none', 'sum', 'mean'])
def test_loss_against_cross_entropy(weight, ignore_index, reduction):
    logits = torch.randn(10, 5)
    target = torch.arange(10) % 4
    
    cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
    CE_loss = cross_entropy(logits, target)
    
    smooth_with_epsilon0 = SmoothLabelCrossEntropyLoss(epsilon=0.0, weight=weight, ignore_index=ignore_index, reduction=reduction)
    SL_loss = smooth_with_epsilon0(logits, target)
    assert (SL_loss - CE_loss).max().item() < 1e-6
    
    focal_with_gamma0 = FocalLoss(gamma=0.0, weight=weight, ignore_index=ignore_index, reduction=reduction)
    FL_loss = focal_with_gamma0(logits, target)
    assert (FL_loss - CE_loss).max().item() < 1e-6



@pytest.mark.parametrize("epsilon", [0.1, 0.2, 0.3])
def test_smooth_label_loss(epsilon):
    logits = torch.zeros(5, 5, dtype=torch.float)
    target = torch.arange(5, dtype=torch.long)
    
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    CE_losses = cross_entropy(logits, target)
    
    smooth = SmoothLabelCrossEntropyLoss(epsilon=epsilon, reduction='none')
    SL_losses = smooth(logits, target)
    
    assert (SL_losses - CE_losses).abs().max().item() < 1e-6



@pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0])
def test_focal_loss(gamma):
    # From badly-classified to well-classified
    logits = torch.stack([torch.arange(-5, 6), -torch.arange(-5, 6)]).T.type(torch.float)
    target = torch.zeros(logits.size(0), dtype=torch.long)
    
    cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    CE_losses = cross_entropy(logits, target)
    
    focal = FocalLoss(gamma=gamma, reduction='none')
    FL_losses = focal(logits, target)
    
    # Cross entropy loss is always larger than focal loss
    # The ratio of CE/FL would be larger if the entry is better-classified
    loss_ratios = (CE_losses / FL_losses).tolist()
    assert all(bad_loss_ratio < well_loss_ratio for bad_loss_ratio, well_loss_ratio in zip(loss_ratios[:-1], loss_ratios[1:]))
    
    