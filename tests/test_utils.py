# -*- coding: utf-8 -*-
import pytest

from eznlp.utils import find_ascending
from eznlp.utils import segment_text_with_hierarchical_seps


@pytest.mark.parametrize("v", [-500, -3, 0, 2, 2.5, 9, 1234.56])
def test_find_ascending(v):
    N = 10
    sequence = list(range(N))
    find, idx = find_ascending(sequence, v)
    sequence.insert(idx, v)
    
    assert find == (v in list(range(N)))
    assert len(sequence) == N + 1
    assert all(sequence[i] <= sequence[i+1] for i in range(N))
    
    
@pytest.mark.parametrize("length, expected_spans", [(0, [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]), 
                                                    (8, [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]), 
                                                    (12, [(0, 10), (10, 15), (15, 25)]), 
                                                    (20, [(0, 15), (15, 25)]), 
                                                    (50, [(0, 25)])])
def test_segment_text(length, expected_spans):
    text = "aaaa,aaaa,aaaa.bbbb,bbbb."
    spans = list(segment_text_with_hierarchical_seps(text, hie_seps=[["\."], [","]], length=length))
    assert spans == expected_spans
    
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(spans[i][1]==spans[i+1][0] for i in range(len(spans)-1))
    
    