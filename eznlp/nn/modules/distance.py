# -*- coding: utf-8 -*-
import torch

from ..functional import multi_rbf_kernels


class MultiKernelMaxMeanDiscrepancyLoss(torch.nn.Module):
    """Multi-kernel maximum mean discrepancy (MK-MMD) loss.

    References
    ----------
    [1] Long et al. Learning Transferable Features with Deep Adaptation Networks. ICML 2015.
    [2] https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
    [3] https://github.com/ZongxianLee/MMD_Loss.Pytorch
    """

    def __init__(
        self, num_kernels: int = 5, multiplier: float = 2.0, sigma: float = None
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.multiplier = multiplier
        self.sigma = sigma

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1/x2: (num1/num2, hid_dim)
        num1, num2 = x1.size(0), x2.size(0)

        x = torch.cat([x1, x2], dim=0)
        # kernels: (num1+num2, num1+num2)
        kernels = multi_rbf_kernels(
            x,
            num_kernels=self.num_kernels,
            multiplier=self.multiplier,
            sigma=self.sigma,
        )

        k11 = kernels[:num1, :num1].mean()
        k12 = kernels[:num1, num1:].mean()
        k21 = kernels[num1:, :num1].mean()
        k22 = kernels[num1:, num1:].mean()
        if num1 == 0 and num2 == 0:
            return torch.tensor(0.0, device=x1.device)
        elif num1 == 0:
            return k22
        elif num2 == 0:
            return k11
        else:
            return k11 - k12 - k21 + k22

    def extra_repr(self):
        return f"num_kernels={self.num_kernels}, multiplier={self.multiplier}, sigma={self.sigma}"
