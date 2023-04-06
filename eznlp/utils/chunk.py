# -*- coding: utf-8 -*-
from typing import List
import re

from ..token import TokenSequence
from . import find_ascending


FLAT = 0       # Flat entities
NESTED = 1     # Nested entities
ARBITRARY = 2  # Arbitrarily overlapping entities


def _is_overlapping(chunk1: tuple, chunk2: tuple):
    # `NESTED` or `ARBITRARY`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def _is_ordered_nested(chunk1: tuple, chunk2: tuple):
    # `chunk1` is nested in `chunk2`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s2 <= s1 and e1 <= e2)


def _is_nested(chunk1: tuple, chunk2: tuple):
    # `NESTED`
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def _is_clashed(chunk1: tuple, chunk2: tuple, allow_level: int=NESTED):
    if allow_level == FLAT:
        return _is_overlapping(chunk1, chunk2)
    elif allow_level == NESTED:
        return _is_overlapping(chunk1, chunk2) and not _is_nested(chunk1, chunk2)
    else:
        return False


def filter_clashed_by_priority(chunks: List[tuple], allow_level: int=NESTED):
    filtered_chunks = []
    for ck in chunks:
        if all(not _is_clashed(ck, ex_ck, allow_level=allow_level) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)
    return filtered_chunks


def detect_overlapping_level(chunks: List[tuple]):
    level = FLAT
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if _is_nested(ck1, ck2):
                level = NESTED
            elif _is_overlapping(ck1, ck2):
                # Non-nested overlapping -> `ARBITRARY`
                return ARBITRARY
    return level


def detect_nested(chunks1: List[tuple], chunks2: List[tuple]=None, strict: bool=True):
    """Return chunks from `chunks1` that are nested in any chunk from `chunks2`. 
    """
    if chunks2 is None:
        chunks2 = chunks1
    
    nested_chunks = []
    for ck1 in chunks1:
        if any(_is_ordered_nested(ck1, ck2) and (ck1 != ck2) and (not strict or ck1[1:] != ck2[1:]) for ck2 in chunks2):
            nested_chunks.append(ck1)
    return nested_chunks

def count_nested(chunks1: List[tuple], chunks2: List[tuple]=None, strict: bool=True):
    return len(detect_nested(chunks1, chunks2=chunks2, strict=strict))


def chunk_pair_distance(chunk1: tuple, chunk2: tuple):
    """Return the distance between two chunks. 
    If the two chunks do not overlap, the distance is the size of span between them.
    If the two chunks overlap (e.g., nested), the distance is the negative size of overlapping span. 
    """
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return max(s1, s2) - min(e1, e2)



class TextChunksTranslator(object):
    """The translator between chunks and text-chunks. 
    
    Args
    ----
    chunks: list
        list of (chunk_type, chunk_start, chunk_end). 
    text_chunks: list
        list of (chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text*), where `chunk_text` is optional
    tokens : `TokenSequence`
        Token sequence object. 
    raw_text : str
        The raw text. 
    """
    def __init__(self, mismatch_tol: int=2, consistency_mapping: dict=None):
        self.mismatch_tol = mismatch_tol
        self.consistency_mapping = {'\s': ' '} 
        if consistency_mapping is not None:
            self.consistency_mapping.update(consistency_mapping)
        
    def is_consistency(self, text: str, gold_text: str):
        if text == gold_text:
            return True
        
        for pattern, repl in self.consistency_mapping.items():
            text = re.sub(pattern, repl, text)
            gold_text = re.sub(pattern, repl, gold_text)
        return text == gold_text
        
        
    def text_chunks2chunks(self, text_chunks: List[tuple], tokens: TokenSequence, raw_text: str=None, place_none_for_errors: bool=False):
        text_starts, text_ends = tokens.start, tokens.end
        
        chunks = []
        errors, mismatches = [], []
        for chunk_type, chunk_start_in_text, chunk_end_in_text, *possible_chunk_text in text_chunks:
            find_start, chunk_start = find_ascending(text_starts, chunk_start_in_text)
            find_end, chunk_end_m1 = find_ascending(text_ends, chunk_end_in_text)
            
            if len(possible_chunk_text) > 0:
                chunk_text = possible_chunk_text[0]
                
                # Error data
                if not self.is_consistency(chunk_text, raw_text[chunk_start_in_text:chunk_end_in_text]):
                    if self.is_consistency(chunk_text, raw_text[chunk_start_in_text:chunk_start_in_text+len(chunk_text)]):
                        chunk_end_in_text = chunk_start_in_text + len(chunk_text)
                    elif self.is_consistency(chunk_text, raw_text[chunk_end_in_text-len(chunk_text):chunk_end_in_text]):
                        chunk_start_in_text = chunk_end_in_text - len(chunk_text)
                    else:
                        errors.append((chunk_text, raw_text[chunk_start_in_text:chunk_end_in_text]))
                        if place_none_for_errors:
                            chunks.append(None)
                        continue
            
            # Mis-matched data
            if not find_start:
                mismatches.append((raw_text[text_starts[chunk_start-1]:text_ends[chunk_start-1]], 
                                   raw_text[text_starts[chunk_start-1]:chunk_start_in_text], 
                                   raw_text[chunk_start_in_text:text_ends[chunk_start-1]]))
            if not find_end:
                mismatches.append((raw_text[text_starts[chunk_end_m1]:text_ends[chunk_end_m1]], 
                                   raw_text[text_starts[chunk_end_m1]:chunk_end_in_text], 
                                   raw_text[chunk_end_in_text:text_ends[chunk_end_m1]]))
            
            # If it is exactly found...
            if find_start and find_end:
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            # If it is almost found... 
            elif (not find_start) and find_end and (chunk_start_in_text - text_starts[chunk_start-1] <= self.mismatch_tol):
                chunk_start = chunk_start - 1
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            elif find_start and (not find_end) and (text_ends[chunk_end_m1] - chunk_end_in_text <= self.mismatch_tol):
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            elif place_none_for_errors:
                chunks.append(None)
        
        return chunks, errors, mismatches
        
        
    def chunks2text_chunks(self, chunks: List[tuple], tokens: TokenSequence, raw_text: str=None, append_chunk_text: bool=False):
        text_starts, text_ends = tokens.start, tokens.end
        
        text_chunks = []
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_start_in_text = text_starts[chunk_start]
            chunk_end_in_text = text_ends[chunk_end-1]
            
            text_chunk = (chunk_type, chunk_start_in_text, chunk_end_in_text)
            if append_chunk_text:
                text_chunk = (*text_chunk, raw_text[chunk_start_in_text:chunk_end_in_text])
            text_chunks.append(text_chunk)
            
        return text_chunks
