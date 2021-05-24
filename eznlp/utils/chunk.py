# -*- coding: utf-8 -*-
from typing import List
import re

from ..token import TokenSequence
from . import find_ascending


def is_overlapped(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 < e2 and s2 < e1)


def is_nested(chunk1: tuple, chunk2: tuple):
    (_, s1, e1), (_, s2, e2) = chunk1, chunk2
    return (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)


def is_clashed(chunk1: tuple, chunk2: tuple, allow_nested: bool=True):
    if allow_nested:
        return is_overlapped(chunk1, chunk2) and not is_nested(chunk1, chunk2)
    else:
        return is_overlapped(chunk1, chunk2)


def filter_clashed_by_priority(chunks: List[tuple], allow_nested: bool=True):
    filtered_chunks = []
    for ck in chunks:
        if all(not is_clashed(ck, ex_ck, allow_nested=allow_nested) for ex_ck in filtered_chunks):
            filtered_chunks.append(ck)
            
    return filtered_chunks


def detect_nested(chunks: List[tuple]):
    for i, ck1 in enumerate(chunks):
        for ck2 in chunks[i+1:]:
            if is_nested(ck1, ck2):
                return True
    return False



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
    def __init__(self, mismatch_tol: int=2):
        self.mismatch_tol = mismatch_tol
        self.mapping = {'\s': ' ', '[-–]': '-'}
        
        
    def text_chunks2chunks(self, text_chunks: List[tuple], tokens: TokenSequence, raw_text: str=None):
        text_starts, text_ends = tokens.start, tokens.end
        
        chunks = []
        errors, mismatches = [], []
        for chunk_type, chunk_start_in_text, chunk_end_in_text, *possible_chunk_text in text_chunks:
            find_start, chunk_start = find_ascending(text_starts, chunk_start_in_text)
            find_end, chunk_end_m1 = find_ascending(text_ends, chunk_end_in_text)
            
            if len(possible_chunk_text) > 0:
                chunk_text = possible_chunk_text[0]
                
                # Error data
                if chunk_text != raw_text[chunk_start_in_text:chunk_end_in_text]:
                    for pattern, repl in self.mapping.items():
                        if re.sub(pattern, repl, chunk_text) == re.sub(pattern, repl, raw_text[chunk_start_in_text:chunk_end_in_text]):
                            break
                        
                    else:
                        if chunk_text == raw_text[chunk_start_in_text:chunk_start_in_text+len(chunk_text)]:
                            chunk_end_in_text = chunk_start_in_text + len(chunk_text)
                        elif chunk_text == raw_text[chunk_end_in_text-len(chunk_text):chunk_end_in_text]:
                            chunk_start_in_text = chunk_end_in_text - len(chunk_text)
                        else:
                            errors.append((chunk_text, raw_text[chunk_start_in_text:chunk_end_in_text]))
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
            if (not find_start) and find_end and (chunk_start_in_text - text_starts[chunk_start-1] <= self.mismatch_tol):
                chunk_start = chunk_start - 1
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            if find_start and (not find_end) and (text_ends[chunk_end_m1] - chunk_end_in_text <= self.mismatch_tol):
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
        
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
