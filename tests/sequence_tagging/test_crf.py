# -*- coding: utf-8 -*-
import torch
import torchcrf

from eznlp.sequence_tagging.crf import CRF


def test_crf(self):
    batch_size = 10
    step = 20
    tag_dim = 5
    emissions = torch.randn(batch_size, step, tag_dim)
    tag_ids = torch.randint(0, tag_dim, (batch_size, step))
    seq_lens = torch.randint(1, step, (batch_size, ))
    mask = (torch.arange(step).unsqueeze(0).repeat(batch_size, 1) >= seq_lens.unsqueeze(-1))
    
    bm_crf = torchcrf.CRF(tag_dim, batch_first=True)
    llh = bm_crf(emissions, tag_ids, (~mask).type(torch.uint8), reduction='none')
    best_paths = bm_crf.decode(emissions, (~mask).type(torch.uint8))
    
    my_crf = CRF(tag_dim, batch_first=True)
    my_crf.sos_transitions.data = bm_crf.start_transitions.data
    my_crf.eos_transitions.data = bm_crf.end_transitions.data
    my_crf.transitions.data = bm_crf.transitions.data
    
    my_losses = my_crf(emissions, tag_ids, mask)
    my_best_paths = my_crf.decode(emissions, mask)
    
    assert (llh + my_losses).abs().max() < 1e-4
    assert my_best_paths == best_paths
    