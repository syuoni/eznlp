# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn.functional import seq_lens2mask
from eznlp.nn import SequenceAttention


@pytest.mark.parametrize("scoring", ['Dot', 'Multiplicative', 'Additive', 'Biaffine'])
def test_sequence_attention(scoring):
    BATCH_SIZE = 100
    MAX_LEN = 20
    HID_DIM = 50
    
    x = torch.randn(BATCH_SIZE, MAX_LEN, HID_DIM)
    seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
    mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
    
    atten_values, atten_weight = SequenceAttention(HID_DIM, scoring=scoring)(x, mask, return_atten_weight=True)
    assert (atten_weight[mask] == 0).all().item()
    assert atten_values.size(0) == BATCH_SIZE
    assert atten_values.size(1) == HID_DIM
