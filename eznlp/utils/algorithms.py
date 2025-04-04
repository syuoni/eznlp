# -*- coding: utf-8 -*-
from typing import List, Union


def find_ascending(
    values: List[Union[int, float]],
    x: Union[int, float],
    start: int = None,
    end: int = None,
):
    """Binary search a value `x` in an ascending sequence `values`.

    Parameters
    ----------
    values : list of values
        A sequence of ascending values.
    x : int, float, etc.
        A value to search.
    start : int, optional
        DESCRIPTION. The default is None.
    end : int, optional
        DESCRIPTION. The default is None.

    Returns: Tuple
    -------
    find: bool
        Whether `value` exists in `sequence`.
    idx: int
        The index of `value` in `sequence`.

    If `find` being True, `sequence[idx] == value`.
    If `find` being False, `sequence[idx-1] < value < sequence[idx]`.
    If calling `sequence.insert(idx, value)`, then `sequence` will remain ascending.
    """
    start = 0 if start is None else start
    end = len(values) if end is None else end

    if end - start <= 0:
        return None, None
    if end - start == 1:
        if values[start] == x:
            return True, start
        elif values[start] < x:
            return False, start + 1
        else:
            return False, start

    mid = start + (end - start) // 2
    if values[mid] <= x:
        return find_ascending(values, x, start=mid, end=end)
    else:
        return find_ascending(values, x, start=start, end=mid)


def _assign_consecutive_to_buckets_min_upper_limit(
    values: List[int], num_buckets: int, start: int, end: int
):
    if end - start == 1:
        return start

    mid = start + (end - start - 1) // 2
    k, curr_sum = 0, 0
    success = True
    for x in values:
        if curr_sum + x <= mid:
            curr_sum += x
        else:
            k += 1
            curr_sum = x
            if k >= num_buckets:
                success = False
                break
    if success:
        return _assign_consecutive_to_buckets_min_upper_limit(
            values, num_buckets, start, mid + 1
        )
    else:
        return _assign_consecutive_to_buckets_min_upper_limit(
            values, num_buckets, mid + 1, end
        )


def assign_consecutive_to_buckets(values: List[int], num_buckets: int):
    min_upper_limit = _assign_consecutive_to_buckets_min_upper_limit(
        values, num_buckets, max(values), sum(values) + 1
    )

    buckets = [0 for _ in range(num_buckets)]
    k, curr_sum = 0, 0
    for x in values:
        if curr_sum + x <= min_upper_limit:
            curr_sum += x
            buckets[k] += 1
        else:
            k += 1
            curr_sum = x
            buckets[k] += 1

    assert all(b > 0 for b in buckets)
    assert sum(buckets) == len(values)
    return buckets
