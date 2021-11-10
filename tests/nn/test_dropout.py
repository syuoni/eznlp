# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn import LockedDropout, WordDropout


@pytest.mark.parametrize("dropout_rate", [0.2, 0.5, 0.8])
def test_locked_dropout(dropout_rate):
    BATCH_SIZE = 100
    MAX_LEN = 200
    HID_DIM = 500
    x = torch.ones(BATCH_SIZE, MAX_LEN, HID_DIM)
    
    dropout = LockedDropout(p=dropout_rate)
    dropout.eval()
    x_locked_dropouted = dropout(x)
    assert (x_locked_dropouted == x).all().item()
    
    dropout.train()
    x_locked_dropouted = dropout(x)
    assert set(x_locked_dropouted.sum(dim=1).long().flatten().tolist()) == {0, int(round(MAX_LEN/(1-dropout_rate)))}
    assert abs(x_locked_dropouted.mean().item() - 1) < 0.05



@pytest.mark.parametrize("dropout_rate", [0.2, 0.5, 0.8])
def test_word_dropout(dropout_rate):
    BATCH_SIZE = 100
    MAX_LEN = 200
    HID_DIM = 500
    x = torch.ones(BATCH_SIZE, MAX_LEN, HID_DIM)
    
    dropout = WordDropout(p=dropout_rate, keep_exp=True)
    dropout.eval()
    x_word_dropouted = dropout(x)
    assert (x_word_dropouted == x).all().item()
    
    dropout.train()
    x_word_dropouted = dropout(x)
    assert set(x_word_dropouted.sum(dim=2).long().flatten().tolist()) == {0, int(round(HID_DIM/(1-dropout_rate)))}
    assert abs(x_word_dropouted.mean().item() - 1) < 0.05
