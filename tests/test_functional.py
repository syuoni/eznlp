# -*- coding: utf-8 -*-
import pytest
import torch
from eznlp.nn.functional import seq_lens2mask, aggregate_tensor_by_group
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
        
        
class TestSeqLens2Mask(object):
    def test_example(self):
        BATCH_SIZE = 100
        MAX_LEN = 20
        seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
        mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
        assert ((MAX_LEN - mask.sum(dim=1)) == seq_lens).all()
        
        for i in range(BATCH_SIZE):
            assert not mask[i, :seq_lens[i]].any()
            assert mask[i, seq_lens[i]:].all()
            
            
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
            
            
            
class TestAggregateTensorByGroup(object):
    def test_example(self):
        BATCH_SIZE = 1
        ORI_STEP = 10
        AGG_STEP = 8
        
        tensor = torch.randn(BATCH_SIZE, ORI_STEP, 20)
        group_by = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 3, -1]])
        agg_tensor = aggregate_tensor_by_group(tensor, group_by, agg_step=AGG_STEP)
        
        agg_tensor_gold = torch.stack([tensor[0][group_by[0] == i].mean(dim=0) for i in range(AGG_STEP)])
        agg_tensor_gold.masked_fill_(agg_tensor_gold.isnan(), 0)
        
        assert (agg_tensor - agg_tensor_gold).abs().max().item() < 1e-6
        
        
    def test_batch(self):
        BATCH_SIZE = 16
        ORI_STEP = 10
        AGG_STEP = 8
        
        tensor = torch.randn(BATCH_SIZE, ORI_STEP, 20)
        group_by = torch.randint(0, AGG_STEP, size=(BATCH_SIZE, ORI_STEP))
        agg_tensor = aggregate_tensor_by_group(tensor, group_by, agg_step=AGG_STEP)
        
        agg_tensor_gold = []
        for k in range(BATCH_SIZE):
            agg_tensor_gold.append(torch.stack([tensor[k][group_by[k] == i].mean(dim=0) for i in range(AGG_STEP)]))
        agg_tensor_gold = torch.stack(agg_tensor_gold)
        agg_tensor_gold.masked_fill_(agg_tensor_gold.isnan(), 0)
        
        assert (agg_tensor - agg_tensor_gold).abs().max().item() < 1e-6

