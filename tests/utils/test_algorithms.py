# -*- coding: utf-8 -*-
import pytest

from eznlp.utils import find_ascending, assign_consecutive_to_buckets


@pytest.mark.parametrize("v", [-500, -3, 0, 2, 2.5, 9, 1234.56])
def test_find_ascending(v):
    N = 10
    sequence = list(range(N))
    find, idx = find_ascending(sequence, v)
    sequence.insert(idx, v)
    
    assert find == (v in list(range(N)))
    assert len(sequence) == N + 1
    assert all(sequence[i] <= sequence[i+1] for i in range(N))



def test_assign_consecutive_to_buckets1():
    N = 10
    values = [5 for _ in range(N)]
    buckets = assign_consecutive_to_buckets(values, N)
    assert buckets == [1 for _ in range(N)]
    buckets = assign_consecutive_to_buckets(values, 1)
    assert buckets == [N]


def test_assign_consecutive_to_buckets2():
    N = 10
    values = list(range(1, N+1))
    buckets = assign_consecutive_to_buckets(values, 7)
    assert buckets == [4, 1, 1, 1, 1, 1, 1]
    buckets = assign_consecutive_to_buckets(values, 5)
    assert buckets == [5, 2, 1, 1, 1]
    buckets = assign_consecutive_to_buckets(values, 4)
    assert buckets == [5, 2, 2, 1]
    buckets = assign_consecutive_to_buckets(values, 3)
    assert buckets == [6, 2, 2]
    buckets = assign_consecutive_to_buckets(values, 2)
    assert buckets == [7, 3]
