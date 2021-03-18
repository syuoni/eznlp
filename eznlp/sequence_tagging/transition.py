# -*- coding: utf-8 -*-
import os
import re
from collections import Counter
import pandas as pd

from ..token import TokenSequence


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



class ChunksTagsTranslator(object):
    """
    The translator between chunks and tags. 
    
    Args
    ----
    tags : list
        list of tags. e.g., ['O', 'B-Ent', 'I-Ent', ...]. 
    chunks: list
        list of (chunk_type, chunk_start, chunk_end). 
        
    text_chunks: list
        list of (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
    raw_text : str
        The raw text. 
    tokens : `TokenSequence`
        Token sequence object. 
        
    References
    ----------
    https://github.com/chakki-works/seqeval
    """
    def __init__(self, scheme='BIOES'):
        assert scheme in ('BIO1', 'BIO2', 'BIOES', 'BMES', 'BILOU', 'OntoNotes')
        self.scheme = scheme
        
        dirname = os.path.dirname(__file__)
        sheet_name = 'BIOES' if scheme in ('BMES', 'BILOU') else scheme
        trans = pd.read_excel(f"{dirname}/transition.xlsx", sheet_name=sheet_name, 
                              usecols=['from_tag', 'to_tag', 'legal', 'end_of_chunk', 'start_of_chunk'])
        
        if scheme in ('BMES', 'BILOU'):
            # Mapping from BIOES to BMES/BILOU
            if scheme == 'BMES':
                mapper = {'B': 'B', 'I': 'M', 'O': 'O', 'E': 'E', 'S': 'S'}
            elif scheme == 'BILOU':
                mapper = {'B': 'B', 'I': 'I', 'O': 'O', 'E': 'L', 'S': 'U'}
            trans['from_tag'] = trans['from_tag'].map(mapper)
            trans['to_tag'] = trans['to_tag'].map(mapper)
            
        trans = trans.set_index(['from_tag', 'to_tag'])
        self.trans = {tr: trans.loc[tr].to_dict() for tr in trans.index.tolist()}
        
    def __repr__(self):
        return f"{self.__class__.__name__}(scheme={self.scheme})"
        
    def check_transitions_legal(self, tags: list):
        """
        Check if the transitions between tags are legal. 
        """
        # TODO: also check types
        padded_tags = ['O'] + [tag.split('-', maxsplit=1)[0] for tag in tags] + ['O']
        return all([self.trans[(prev_tag, this_tag)]['legal'] for prev_tag, this_tag in zip(padded_tags[:-1], padded_tags[1:])])
        
    
    def chunks2group_by(self, chunks: list, seq_len: int):
        group_by = [-1 for _ in range(seq_len)]
        
        for i, (chunk_type, chunk_start, chunk_end) in enumerate(chunks):
            for j in range(chunk_start, chunk_end):
                group_by[j] = i
                
        return group_by
        
        
    def chunks2tags(self, chunks: list, seq_len: int):
        tags = ['O' for _ in range(seq_len)]
        
        # Longer chunks are of higher priority
        chunks = sorted(chunks, key=lambda ck: ck[2]-ck[1], reverse=True)
        
        for chunk_type, chunk_start, chunk_end in chunks:
            # Make sure the target slice is contained in the sequence
            # E.g., the sequence is truncated from a longer one
            # This causes unretrievable chunks
            if chunk_start < 0 or chunk_end > seq_len:
                continue
            
            # Make sure the target slice has not been labeled
            # E.g., nested entities
            # This causes unretrievable chunks
            if not all(tags[k] == 'O' for k in range(chunk_start, chunk_end)):
                continue
            
            if self.scheme == 'BIO1':
                if chunk_start == 0 or tags[chunk_start-1] == 'O':
                    tags[chunk_start] = 'I-' + chunk_type
                else:
                    tags[chunk_start] = 'B-' + chunk_type
                for k in range(chunk_start+1, chunk_end):
                    tags[k] = 'I-' + chunk_type
                if chunk_end < len(tags) and tags[chunk_end].startswith('I-'):
                    tags[chunk_end] = tags[chunk_end].replace('I-', 'B-', 1)
                    
            elif self.scheme == 'BIO2':
                tags[chunk_start] = 'B-' + chunk_type
                for k in range(chunk_start+1, chunk_end):
                    tags[k] = 'I-' + chunk_type
                    
            elif self.scheme == 'BIOES':
                if chunk_end - chunk_start == 1:
                    tags[chunk_start] = 'S-' + chunk_type
                else:
                    tags[chunk_start] = 'B-' + chunk_type
                    tags[chunk_end-1] = 'E-' + chunk_type
                    for k in range(chunk_start+1, chunk_end-1):
                        tags[k] = 'I-' + chunk_type
            
            elif self.scheme == 'OntoNotes':
                if chunk_end - chunk_start == 1:
                    tags[chunk_start] = '(' + chunk_type + ')'
                else:
                    tags[chunk_start] = '(' + chunk_type + '*'
                    tags[chunk_end-1] = '*)'
                    for k in range(chunk_start+1, chunk_end-1):
                        tags[k] = '*'
                        
        if self.scheme == 'OntoNotes':
            tags = ['*' if tag == 'O' else tag for tag in tags]
            
        return tags
        
    
    def _vote_in_types(self, chunk_types, breaking_for_types: bool=True):
        if breaking_for_types:
            # All elements must be the same. 
            return chunk_types[0]
        else:
            # Assign the first element with 0.5 higher weight. 
            type_counter = Counter(chunk_types)
            type_counter[chunk_types[0]] += 0.5
            return type_counter.most_common(1)[0][0]
    
    def tags2chunks(self, tags: list, breaking_for_types: bool=True):
        if self.scheme == 'OntoNotes':
            return self.ontonotes_tags2chunks(tags)
        
        chunks = []
        prev_tag, prev_type = 'O', 'O'
        chunk_start, chunk_types = -1, []
        
        for k, tag in enumerate(tags):
            if tag in ('O', '<pad>'):
                this_tag, this_type = 'O', 'O'
            else:
                if '-' in tag:
                    this_tag, this_type = tag.split('-', maxsplit=1)
                else:
                    # Typically cascade-tags without types
                    this_tag, this_type = tag, '<pseudo-type>'
                    
            this_trans = self.trans[(prev_tag, this_tag)]
            is_in_chunk = (prev_tag != 'O') and (this_tag != 'O') and (not this_trans['end_of_chunk']) and (not this_trans['start_of_chunk'])
            
            # Breaking because of different types, is holding only in case of `is_in_chunk` being True. 
            # In such case, the `prev_tag` must be `B` or `I` and 
            #               the `this_tag` must be `I` or `E`. 
            # The breaking operation is equivalent to treating `this_tag` as `B`. 
            if is_in_chunk and breaking_for_types and (this_type != prev_type):
                this_trans = self.trans[(prev_tag, 'B')]
                is_in_chunk = False
                
            if this_trans['end_of_chunk']:
                chunks.append((self._vote_in_types(chunk_types, breaking_for_types), chunk_start, k))
                chunk_types = []
                
            if this_trans['start_of_chunk']:
                chunk_start = k
                chunk_types = [this_type]
                
            if is_in_chunk:
                chunk_types.append(this_type)
                
            prev_tag, prev_type = this_tag, this_type
            
            
        if prev_tag != 'O':
            chunks.append((self._vote_in_types(chunk_types, breaking_for_types), chunk_start, len(tags)))
            
        return chunks
        
    def ontonotes_tags2chunks(self, tags: list):
        chunks = []
        prev_tag = '*)'
        chunk_start, chunk_type = -1, None
        
        for k, tag in enumerate(tags):
            this_tag = "".join(re.findall('[\(\*\)]', tag))
            this_type = re.sub('[\(\*\)]', '', tag)
            
            this_trans = self.trans[(prev_tag, this_tag)]
            
            if this_trans['end_of_chunk'] and (chunk_type is not None):
                chunks.append((chunk_type, chunk_start, k))
                chunk_type = None
                
            if this_trans['start_of_chunk']:
                chunk_start = k
                chunk_type = this_type
                
            prev_tag = this_tag
            
            
        if self.trans[(prev_tag, '(*')]['end_of_chunk']:
            chunks.append((chunk_type, chunk_start, len(tags)))
            
        return chunks
    
        
    def chunks2text_chunks(self, chunks: list, raw_text: str, tokens: TokenSequence):
        text_starts, text_ends = tokens.start, tokens.end
        
        text_chunks = []
        for chunk_type, chunk_start, chunk_end in chunks:
            chunk_start_in_text = text_starts[chunk_start]
            chunk_end_in_text = text_ends[chunk_end-1]
            text_chunks.append((raw_text[chunk_start_in_text:chunk_end_in_text], 
                                chunk_type, 
                                chunk_start_in_text, 
                                chunk_end_in_text))
        return text_chunks
    
        
    def text_chunks2chunks(self, text_chunks: list, raw_text: str, tokens: TokenSequence, mismatch_tol: int=2):
        text_starts, text_ends = tokens.start, tokens.end
        
        chunks = []
        errors, mismatches = [], []
        for chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text in text_chunks:
            # Error data
            if chunk_text != raw_text[chunk_start_in_text:chunk_end_in_text]:
                if re.sub('[-–]', '-', chunk_text) == re.sub('[-–]', '-', raw_text[chunk_start_in_text:chunk_end_in_text]):
                    pass
                elif re.sub('\s', ' ', chunk_text) == re.sub('\s', ' ', raw_text[chunk_start_in_text:chunk_end_in_text]):
                    pass
                elif chunk_text == raw_text[chunk_start_in_text:chunk_start_in_text+len(chunk_text)]:
                    chunk_end_in_text = chunk_start_in_text + len(chunk_text)
                elif chunk_text == raw_text[chunk_end_in_text-len(chunk_text):chunk_end_in_text]:
                    chunk_start_in_text = chunk_end_in_text - len(chunk_text)
                else:
                    errors.append((chunk_text, raw_text[chunk_start_in_text:chunk_end_in_text]))
                    continue
                
            find_start, chunk_start = find_ascending(text_starts, chunk_start_in_text)
            find_end, chunk_end_m1 = find_ascending(text_ends, chunk_end_in_text)
            
            # Mis-matched data
            if not find_start:
                mismatches.append([raw_text[text_starts[chunk_start-1]:text_ends[chunk_start-1]], 
                                   raw_text[text_starts[chunk_start-1]:chunk_start_in_text], 
                                   raw_text[chunk_start_in_text:text_ends[chunk_start-1]]])
            if not find_end:
                mismatches.append([raw_text[text_starts[chunk_end_m1]:text_ends[chunk_end_m1]], 
                                   raw_text[text_starts[chunk_end_m1]:chunk_end_in_text], 
                                   raw_text[chunk_end_in_text:text_ends[chunk_end_m1]]])
            
            # If it is exactly found...
            if find_start and find_end:
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            # If it is almost found... 
            if (not find_start) and find_end and (chunk_start_in_text - text_starts[chunk_start-1] <= mismatch_tol):
                chunk_start = chunk_start - 1
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
                
            if find_start and (not find_end) and (text_ends[chunk_end_m1] - chunk_end_in_text <= mismatch_tol):
                chunks.append((chunk_type, chunk_start, chunk_end_m1+1))
        
        return chunks, errors, mismatches
    
    
    def tags2text_chunks(self, tags: list, raw_text: str, tokens: TokenSequence, **kwargs):
        chunks = self.tags2chunks(tags, **kwargs)
        return self.chunks2text_chunks(chunks, raw_text, tokens)
        
    
    def text_chunks2tags(self, text_chunks: list, raw_text: str, tokens: TokenSequence, **kwargs):
        chunks, errors, mismatches = self.text_chunks2chunks(text_chunks, raw_text, tokens, **kwargs)
        return self.chunks2tags(chunks, len(tokens), **kwargs), errors, mismatches
    
    