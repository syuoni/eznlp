# -*- coding: utf-8 -*-
import pytest
import torch

@pytest.fixture
def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
