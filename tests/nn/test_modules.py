# -*- coding: utf-8 -*-
import pytest
import torch
from eznlp.nn.functional import seq_lens2mask
from eznlp.nn import MaxPooling, MeanPooling
from eznlp.nn import LockedDropout, WordDropout


class TestDropout(object):
    @pytest.mark.parametrize("dropout_rate", [0.2, 0.5, 0.8])
    def test_locked_dropout(self, dropout_rate):
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
        assert set(x_locked_dropouted.sum(dim=1).type(torch.long).flatten().tolist()) == {0, int(round(MAX_LEN/(1-dropout_rate)))}
        assert abs(x_locked_dropouted.mean().item() - 1) < 0.05
        
    @pytest.mark.parametrize("dropout_rate", [0.2, 0.5, 0.8])
    def test_word_dropout(self, dropout_rate):
        BATCH_SIZE = 100
        MAX_LEN = 200
        HID_DIM = 500
        x = torch.ones(BATCH_SIZE, MAX_LEN, HID_DIM)
        
        dropout = WordDropout(p=dropout_rate)
        dropout.eval()
        x_word_dropouted = dropout(x)
        assert (x_word_dropouted == x).all().item()
        
        dropout.train()
        x_word_dropouted = dropout(x)
        assert set(x_word_dropouted.sum(dim=2).type(torch.long).flatten().tolist()) == {0, int(round(HID_DIM/(1-dropout_rate)))}
        assert abs(x_word_dropouted.mean().item() - 1) < 0.05
        
        
            
class TestPooling(object):
    def test_pooling(self):
        BATCH_SIZE = 100
        MAX_LEN = 20
        HID_DIM = 50
        
        x = torch.randn(BATCH_SIZE, MAX_LEN, HID_DIM)
        seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
        mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
        
        max_pooled  = MaxPooling()(x, mask)
        mean_pooled = MeanPooling()(x, mask)
        
        for i in range(BATCH_SIZE):
            assert (max_pooled[i]  - x[i, :seq_lens[i]].max(dim=0).values).abs().max().item() < 1e-6
            assert (mean_pooled[i] - x[i, :seq_lens[i]].mean(dim=0)).abs().max().item() < 1e-6
            
            
        
        
        