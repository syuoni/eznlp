# -*- coding: utf-8 -*-
import pytest
import os
import torch

from eznlp.model import ELMoConfig
from eznlp.training import count_params


@pytest.mark.parametrize("mix_layers", ['trainable', 'top', 'average'])
@pytest.mark.parametrize("use_gamma", [True, False])
@pytest.mark.parametrize("freeze", [True, False])
def test_trainble_config(mix_layers, use_gamma, freeze, elmo):
    elmo_config = ELMoConfig(elmo=elmo, freeze=freeze, mix_layers=mix_layers, use_gamma=use_gamma)
    elmo_embedder = elmo_config.instantiate()
    
    expected_num_trainable_params = 0
    if not freeze:
        expected_num_trainable_params += count_params(elmo, return_trainable=False) - 4
    if mix_layers.lower() == 'trainable':
        expected_num_trainable_params += 3
    if use_gamma:
        expected_num_trainable_params += 1
    
    assert count_params(elmo_embedder) == expected_num_trainable_params



def test_serialization(elmo):
    config = ELMoConfig(elmo=elmo)
    
    config_path = "cache/elmo_embedder.config"
    torch.save(config, config_path)
    assert os.path.getsize(config_path) < 1024 * 1024  # 1MB
