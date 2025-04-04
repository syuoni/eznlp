# -*- coding: utf-8 -*-
import pytest
import torch

import third_party.transferlearning.code.distance
from eznlp.nn import MultiKernelMaxMeanDiscrepancyLoss


class TestMKMMDLoss(object):
    @pytest.mark.parametrize("num_kernels", [5, 10])
    @pytest.mark.parametrize("multiplier", [2.0, 5.0])
    def test_against_third_party(self, num_kernels, multiplier):
        x1 = torch.randn(20, 100)
        x2 = torch.randn(20, 100)
        benchmark_mmd = third_party.transferlearning.code.distance.MMD_loss(kernel_num=num_kernels, kernel_mul=multiplier)
        benchmark_res = benchmark_mmd(x1, x2)

        mmd = MultiKernelMaxMeanDiscrepancyLoss(num_kernels=num_kernels, multiplier=multiplier)
        mmd_res = mmd(x1, x2)
        assert (mmd_res - benchmark_res).abs().max().item() < 1e-5
