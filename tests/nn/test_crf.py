# -*- coding: utf-8 -*-
import torch
import torchcrf

from eznlp.nn import CRF


def test_crf():
    batch_size = 10
    step = 20
    tag_dim = 5
    emissions = torch.randn(batch_size, step, tag_dim)
    tag_ids = torch.randint(0, tag_dim, (batch_size, step))
    seq_lens = torch.randint(1, step, (batch_size, ))
    mask = (torch.arange(step).unsqueeze(0).expand(batch_size, -1) >= seq_lens.unsqueeze(-1))
    
    benchmark_crf = torchcrf.CRF(tag_dim, batch_first=True)
    benchmark_llh = benchmark_crf(emissions, tag_ids, (~mask).type(torch.uint8), reduction='none')
    benchmark_best_paths = benchmark_crf.decode(emissions, (~mask).type(torch.uint8))
    
    crf = CRF(tag_dim, batch_first=True)
    crf.sos_transitions.data = benchmark_crf.start_transitions.data
    crf.eos_transitions.data = benchmark_crf.end_transitions.data
    crf.transitions.data = benchmark_crf.transitions.data
    
    losses = crf(emissions, tag_ids, mask)
    best_paths = crf.decode(emissions, mask)
    
    assert (benchmark_llh + losses).abs().max() < 1e-4
    assert best_paths == benchmark_best_paths
    