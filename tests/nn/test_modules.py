# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn.functional import seq_lens2mask
from eznlp.nn import SequencePooling, SequenceAttention, SequenceGroupAggregating
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
        
        dropout = WordDropout(p=dropout_rate, keep_exp=True)
        dropout.eval()
        x_word_dropouted = dropout(x)
        assert (x_word_dropouted == x).all().item()
        
        dropout.train()
        x_word_dropouted = dropout(x)
        assert set(x_word_dropouted.sum(dim=2).type(torch.long).flatten().tolist()) == {0, int(round(HID_DIM/(1-dropout_rate)))}
        assert abs(x_word_dropouted.mean().item() - 1) < 0.05




class TestSequencePooling(object):
    @pytest.mark.parametrize("mode", ['Mean', 'Max', 'Min'])
    def test_pooling(self, mode):
        BATCH_SIZE = 100
        MAX_LEN = 20
        HID_DIM = 50
        
        x = torch.randn(BATCH_SIZE, MAX_LEN, HID_DIM)
        seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
        mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
        
        pooled = SequencePooling(mode=mode)(x, mask)
        
        for i in range(BATCH_SIZE):
            if mode.lower() == 'mean':
                assert (pooled[i] - x[i, :seq_lens[i]].mean(dim=0)).abs().max().item() < 1e-6
            elif mode.lower() == 'max':
                assert (pooled[i] - x[i, :seq_lens[i]].max(dim=0).values).abs().max().item() < 1e-6
            elif mode.lower() == 'min':
                assert (pooled[i] - x[i, :seq_lens[i]].min(dim=0).values).abs().max().item() < 1e-6
    
    
    @pytest.mark.parametrize("scoring", ['Dot', 'Multiplicative', 'Additive'])
    def test_attention(self, scoring):
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




class TestAggregateTensorByGroup(object):
    @pytest.mark.parametrize("x, group_by", [(torch.randn(1, 10, 20), 
                                              torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 3, -1]])), 
                                             (torch.randn(16, 10, 20), 
                                              torch.randint(0, 8, size=(16, 10)))])
    @pytest.mark.parametrize("agg_mode", ['mean', 'max', 'min', 'first', 'last'])
    def test_example(self, x, group_by, agg_mode):
        AGG_STEP = 8
        agg_tensor = SequenceGroupAggregating(mode=agg_mode)(x, group_by, agg_step=AGG_STEP)
        
        agg_tensor_gold = []
        for k in range(x.size(0)):
            curr_agg_tensor_gold = []
            for i in range(AGG_STEP):
                piece = x[k][group_by[k] == i]
                if piece.size(0) > 0:
                    if agg_mode.lower() == 'mean':
                        curr_agg_tensor_gold.append(piece.mean(dim=0))
                    elif agg_mode.lower() == 'max':
                        curr_agg_tensor_gold.append(piece.max(dim=0).values)
                    elif agg_mode.lower() == 'min':
                        curr_agg_tensor_gold.append(piece.min(dim=0).values)
                    elif agg_mode.lower() == 'first':
                        curr_agg_tensor_gold.append(piece[0])
                    elif agg_mode.lower() == 'last':
                        curr_agg_tensor_gold.append(piece[-1])
                else:
                    curr_agg_tensor_gold.append(torch.zeros(piece.size(1)))
            agg_tensor_gold.append(torch.stack(curr_agg_tensor_gold))
        
        agg_tensor_gold = torch.stack(agg_tensor_gold)
        assert (agg_tensor - agg_tensor_gold).abs().max().item() < 1e-6
        