# -*- coding: utf-8 -*-


def find_ascending(sequence: list, value, start: int=None, end: int=None):
    """
    Binary search `value` in `sequence` which is assumed to be ascending. 
    
    Parameters
    ----------
    sequence : list of values
        A sequence of ascending values. 
    value : int, float, etc. 
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
    end = len(sequence) if end is None else end
    if start >= end:
        return None, None
    elif start + 1 == end:
        if sequence[start] == value:
            return True, start
        elif sequence[start] < value:
            return False, start + 1
        else:
            return False, start
    
    mid = (start + end) // 2
    if sequence[mid] <= value:
        return find_ascending(sequence, value, start=mid, end=end)
    else:
        return find_ascending(sequence, value, start=start, end=mid)
