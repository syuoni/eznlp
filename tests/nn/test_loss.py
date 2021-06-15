# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn import SmoothLabelCrossEntropyLoss, FocalLoss
import third_party.dice_loss_for_NLP.loss


class TestFocalLoss(object):
    @pytest.mark.parametrize("weight", [None, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])])
    @pytest.mark.parametrize("ignore_index", [-100, 0])
    @pytest.mark.parametrize("reduction", ['none', 'sum', 'mean'])
    def test_with_gamma0(self, weight, ignore_index, reduction):
        logits = torch.randn(10, 5)
        target = torch.arange(10) % 4
        
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        CE_logits = logits.clone().detach().requires_grad_(True)
        CE_loss = cross_entropy(CE_logits, target)
        
        focal_with_gamma0 = FocalLoss(gamma=0.0, weight=weight, ignore_index=ignore_index, reduction=reduction)
        FL_logits = logits.clone().detach().requires_grad_(True)
        FL_loss = focal_with_gamma0(FL_logits, target)
        assert (FL_loss - CE_loss).abs().max().item() < 1e-5
        
        if reduction != 'none':
            CE_loss.backward()
            FL_loss.backward()
            assert (FL_logits.grad - CE_logits.grad).abs().max().item() < 1e-6
        
        
    @pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0])
    def test_loss_value_against_CE(self, gamma):
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
        
        
    @pytest.mark.parametrize("weight", [None, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])])
    @pytest.mark.parametrize("reduction", ['none', 'sum'])
    @pytest.mark.parametrize("gamma", [1.0, 2.0, 3.0])
    def test_against_third_party(self, weight, reduction, gamma):
        logits = torch.randn(10, 5)
        target = torch.arange(10) % 4
        
        benchmark_focal = third_party.dice_loss_for_NLP.loss.FocalLoss(gamma=gamma, alpha=weight.tolist() if weight is not None else None, reduction=reduction)
        BM_logits = logits.clone().detach().requires_grad_(True)
        BM_loss = benchmark_focal(BM_logits, target)
        
        focal = FocalLoss(gamma=gamma, weight=weight, reduction=reduction)
        FL_logits = logits.clone().detach().requires_grad_(True)
        FL_loss = focal(FL_logits, target)
        assert (FL_loss - BM_loss).abs().max().item() < 1e-5
        
        if reduction != 'none':
            BM_loss.backward()
            FL_loss.backward()
            assert (FL_logits.grad - BM_logits.grad).abs().max().item() < 1e-6




class TestSmoothLabel(object):
    @pytest.mark.parametrize("weight", [None, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])])
    @pytest.mark.parametrize("ignore_index", [-100, 0])
    @pytest.mark.parametrize("reduction", ['none', 'sum', 'mean'])
    def test_with_epsilon0(self, weight, ignore_index, reduction):
        logits = torch.randn(10, 5)
        target = torch.arange(10) % 4
        
        cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        CE_logits = logits.clone().detach().requires_grad_(True)
        CE_loss = cross_entropy(CE_logits, target)
        
        smooth_with_epsilon0 = SmoothLabelCrossEntropyLoss(epsilon=0.0, weight=weight, ignore_index=ignore_index, reduction=reduction)
        SL_logits = logits.clone().detach().requires_grad_(True)
        SL_loss = smooth_with_epsilon0(SL_logits, target)
        assert (SL_loss - CE_loss).abs().max().item() < 1e-5
        
        if reduction != 'none':
            CE_loss.backward()
            SL_loss.backward()
            assert (SL_logits.grad - CE_logits.grad).abs().max().item() < 1e-6
        
        
    @pytest.mark.parametrize("epsilon", [0.1, 0.2, 0.3])
    def test_loss_value_against_CE(self, epsilon):
        logits = torch.zeros(5, 5, dtype=torch.float)
        target = torch.arange(5, dtype=torch.long)
        
        cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        CE_losses = cross_entropy(logits, target)
        
        smooth = SmoothLabelCrossEntropyLoss(epsilon=epsilon, reduction='none')
        SL_losses = smooth(logits, target)
        
        assert (SL_losses - CE_losses).abs().max().item() < 1e-6
