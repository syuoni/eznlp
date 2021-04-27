# -*- coding: utf-8 -*-
import torch

from eznlp.nn.functional import seq_lens2mask, mask2seq_lens


def test_seq_lens2mask():
    BATCH_SIZE = 100
    MAX_LEN = 20
    seq_lens = torch.randint(0, MAX_LEN, size=(BATCH_SIZE, )) + 1
    mask = seq_lens2mask(seq_lens, max_len=MAX_LEN)
    
    assert ((MAX_LEN - mask.sum(dim=1)) == seq_lens).all().item()
    for i in range(BATCH_SIZE):
        assert not mask[i, :seq_lens[i]].any().item()
        assert mask[i, seq_lens[i]:].all().item()
        
    seq_lens_retr = mask2seq_lens(mask)
    assert (seq_lens_retr == seq_lens).all().item()

