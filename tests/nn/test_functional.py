# -*- coding: utf-8 -*-
import pytest
import torch
from eznlp.nn.functional import seq_lens2mask, aggregate_tensor_by_group

        
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
            
            
            
class TestAggregateTensorByGroup(object):
    @pytest.mark.parametrize("x, group_by", [(torch.randn(1, 10, 20), 
                                              torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 3, -1]])), 
                                             (torch.randn(16, 10, 20), 
                                              torch.randint(0, 8, size=(16, 10)))])
    @pytest.mark.parametrize("agg_mode", ['mean', 'max', 'min', 'first', 'last'])
    def test_example(self, x, group_by, agg_mode):
        AGG_STEP = 8
        agg_tensor = aggregate_tensor_by_group(x, group_by, agg_mode=agg_mode, agg_step=AGG_STEP)
        
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
        
        
        
        