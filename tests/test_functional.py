# -*- coding: utf-8 -*-
import torch
from eznlp.functional import seq_lens2mask
from eznlp.functional import max_pooling, mean_pooling
from eznlp.functional import aggregate_tensor_by_group


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
        
        max_pooled  = max_pooling(x, mask)
        mean_pooled = mean_pooling(x, mask)
        
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

