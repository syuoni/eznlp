# -*- coding: utf-8 -*-
from typing import List
import re


def find_ascending(sequence: list, value, start=None, end=None):
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




def segment_text_with_seps(text: str, seps: List[str], length: int=0):
    """
    Segment text with a list of separators. 
    
    If `length` provided, the resulting spans will be as long as possible to be close to `length`. 
    A span may exceed `length`, if the corresponding distance between two successive separators exceeds it. 
    """
    start, end = 0, None
    for cut in re.finditer("|".join(seps), text):
        if cut.end() - start <= length:
            end = cut.end()
            
        elif end is None:
            yield (start, cut.end())
            start = cut.end()
            
        else:
            yield (start, end)
            
            if cut.end() - end <= length:
                start, end = end, cut.end()
            else:
                yield (end, cut.end())
                start, end = cut.end(), None
                
    if len(text) > start:
        if len(text) - start <= length or end is None:
            yield (start, len(text))
        else:
            yield (start, end)
            yield (end, len(text))


def segment_text_with_hierarchical_seps(text: str, hie_seps: List[List[str]], length: int=0):
    """
    Segment text first with seperators in `hie_seps[0]`. For the spans longer than `length`, 
    further segment the spans with separators in `hie_seps[1]`, and so on. 
    """
    if len(hie_seps) == 0:
        yield (0, len(text))
        
    else:
        for start, end in segment_text_with_seps(text, hie_seps[0], length=length):
            if end - start <= length:
                yield (start, end)
            else:
                for sub_start, sub_end in segment_text_with_hierarchical_seps(text[start:end], hie_seps[1:], length=length):
                    # Add offset to the spans from sub-spans
                    yield (start+sub_start, start+sub_end)


def segment_text_uniformly(text: str, num_spans: int=None, max_span_size: int=None):
    assert not (num_spans is None and max_span_size is None)
    
    if num_spans is None:
        num_spans, tail = divmod(len(text), max_span_size)
        if tail > 0:
            num_spans += 1
        
    span_size = len(text) / num_spans
    for i in range(num_spans):
        start = int(span_size* i    + 0.5)
        end   = int(span_size*(i+1) + 0.5)
        yield (start, end)


