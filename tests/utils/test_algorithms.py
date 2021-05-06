# -*- coding: utf-8 -*-
import pytest

from eznlp.utils import find_ascending


@pytest.mark.parametrize("v", [-500, -3, 0, 2, 2.5, 9, 1234.56])
def test_find_ascending(v):
    N = 10
    sequence = list(range(N))
    find, idx = find_ascending(sequence, v)
    sequence.insert(idx, v)
    
    assert find == (v in list(range(N)))
    assert len(sequence) == N + 1
    assert all(sequence[i] <= sequence[i+1] for i in range(N))
