# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn import MultiAffineFusor


@pytest.mark.parametrize("num_affines", [2, 3, 4])
def test_affine_fusor(num_affines):
    BATCH_SIZE = 4
    MAX_LEN = 20
    IN_DIM = 50
    OUT_DIM = 5

    fusor = MultiAffineFusor(num_affines, IN_DIM, OUT_DIM)
    fused = fusor(*(torch.randn(IN_DIM) for _ in range(num_affines)))
    assert fused.size() == (OUT_DIM,)

    fused = fusor(
        *(torch.randn(BATCH_SIZE, MAX_LEN, IN_DIM) for _ in range(num_affines))
    )
    assert fused.size() == (BATCH_SIZE, MAX_LEN, OUT_DIM)
    assert abs(fused.std().item() - 1) < 0.1


def test_affine_fusor_broadcast():
    BATCH_SIZE = 4
    MAX_LEN = 20
    IN_DIM = 50
    OUT_DIM = 5

    fusor = MultiAffineFusor(4, IN_DIM, OUT_DIM)
    fused = fusor(
        torch.randn(BATCH_SIZE, MAX_LEN, IN_DIM),
        torch.randn(BATCH_SIZE, 1, IN_DIM),
        torch.randn(1, MAX_LEN, IN_DIM),
        torch.randn(1, 1, IN_DIM),
    )
    assert fused.size() == (BATCH_SIZE, MAX_LEN, OUT_DIM)
