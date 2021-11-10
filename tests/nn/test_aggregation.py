# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.nn.functional import seq_lens2mask
from eznlp.nn import SequencePooling, SequenceGroupAggregating


@pytest.mark.parametrize("mode, f_agg", [('mean', lambda x: x.mean(dim=0)), 
                                         ('max',  lambda x: x.max(dim=0).values), 
                                         ('min',  lambda x: x.min(dim=0).values)])
def test_sequence_pooling(mode, f_agg):
    BATCH_SIZE = 100
    MAX_LEN = 20
    HID_DIM = 50
    
    x = torch.randn(BATCH_SIZE, MAX_LEN, HID_DIM)
    seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
    mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
    
    pooled = SequencePooling(mode=mode)(x, mask)
    
    for i in range(BATCH_SIZE):
        assert (pooled[i] - f_agg(x[i, :seq_lens[i]])).abs().max().item() < 1e-6



@pytest.mark.parametrize("use_mask", [True, False])
def test_rnn_last(use_mask):
    BATCH_SIZE = 100
    MAX_LEN = 20
    HID_DIM = 50
    
    rnn = torch.nn.LSTM(HID_DIM, HID_DIM, num_layers=2, bidirectional=True, batch_first=True)
    x = torch.randn(BATCH_SIZE, MAX_LEN, HID_DIM)
    
    if use_mask:
        seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
        mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
        
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        packed_y, (hidden, _) = rnn(packed_x)
        y, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_y, batch_first=True, padding_value=0)
    else:
        mask = None
        y, (hidden, _) = rnn(x)
    
    # The last hidden states on the top layer
    last_hidden_gold = torch.cat([hidden[-2], hidden[-1]], dim=-1)
    last_hidden_agg  = SequencePooling(mode='rnn_last')(y, mask)
    assert (last_hidden_agg == last_hidden_gold).all().item()




@pytest.mark.parametrize("x, group_by", [(torch.randn(1, 10, 20), 
                                          torch.tensor([[0, 0, 1, 1, 2, 3, 3, 3, 3, -1]])), 
                                         (torch.randn(16, 10, 20), 
                                          torch.randint(0, 8, size=(16, 10)))])
@pytest.mark.parametrize("agg_mode, f_agg", [('mean',  lambda x: x.mean(dim=0)), 
                                             ('max',   lambda x: x.max(dim=0).values), 
                                             ('min',   lambda x: x.min(dim=0).values), 
                                             ('first', lambda x: x[0]), 
                                             ('last',  lambda x: x[-1])])
def test_sequence_group_aggregating(x, group_by, agg_mode, f_agg):
    AGG_STEP = 8
    agg_tensor = SequenceGroupAggregating(mode=agg_mode)(x, group_by, agg_step=AGG_STEP)
    
    agg_tensor_gold = []
    for k in range(x.size(0)):
        curr_agg_tensor_gold = []
        for i in range(AGG_STEP):
            piece = x[k][group_by[k] == i]
            if piece.size(0) > 0:
                curr_agg_tensor_gold.append(f_agg(piece))
            else:
                curr_agg_tensor_gold.append(torch.zeros(piece.size(1)))
        agg_tensor_gold.append(torch.stack(curr_agg_tensor_gold))
    
    agg_tensor_gold = torch.stack(agg_tensor_gold)
    assert (agg_tensor - agg_tensor_gold).abs().max().item() < 1e-6
