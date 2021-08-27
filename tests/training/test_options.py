# -*- coding: utf-8 -*-
import pytest
import numpy

from eznlp.training import OptionSampler

@pytest.mark.parametrize("lr_range, num_lrs", [(numpy.logspace(-3, -5, num=9, base=10).tolist(), 9), 
                                               (numpy.logspace(-3, -5, num=99, base=10).tolist(), 99)])
@pytest.mark.parametrize("num_options", [10, 20, 50])
def test_option_sampler(lr_range, num_lrs, num_options):
    sampler = OptionSampler(num_epochs=[50, 100], 
                            batch_size=32,
                            lr=lr_range, 
                            fs=None, 
                            use_interm2=[True, False], 
                            bert_arch=['BERT', 'RoBERTa'],)
    assert sampler.num_possible_options == num_lrs*8
    
    full_options = sampler.fully_sample()
    assert len(full_options) == num_lrs*8
    
    random_options = sampler.randomly_sample(num_options)
    assert len(random_options) == num_options
    
    even_options = sampler.evenly_sample(num_options)
    if num_options > sampler.num_possible_options*0.25:
        assert len(even_options) <= num_options
    else:
        assert len(even_options) == num_options
