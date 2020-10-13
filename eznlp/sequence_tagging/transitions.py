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
            return False, start+1
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
    def __init__(self, labeling='BIOES'):
        assert labeling in ('BIO', 'BIOES')
        self.labeling = labeling
        
        dirname = os.path.dirname(__file__)
        trans = pd.read_excel(f"{dirname}/transitions.xlsx", labeling, index_col=[0, 1])
        self.trans = {tr: trans.loc[tr].to_dict() for tr in trans.index.tolist()}
        
        
    def check_transitions_legal(self, tags: list):
        """
        Check if the transitions between tags are legal. 
        """
        # TODO: also check types
        padded_tags = ['O'] + [tag.split('-', maxsplit=1)[0] for tag in tags] + ['O']
        return all([self.trans[(prev_tag, this_tag)]['legal'] for prev_tag, this_tag in zip(padded_tags[:-1], padded_tags[1:])])

        
    def chunks2tags(self, chunks: list, seq_len: int):
        tags = ['O' for _ in range(seq_len)]
        
        # Longer chunks are of higher priority
        chunks = sorted(chunks, key=lambda ck: ck[2]-ck[1], reverse=True)
        
        for chunk_type, chunk_start, chunk_end in chunks:
            if not all(tags[k] == 'O' for k in range(chunk_start, chunk_end)):
                continue
            
            if self.labeling == 'BIO':
                tags[chunk_start] = 'B-' + chunk_type
                for k in range(chunk_start+1, chunk_end):
                    tags[k] = 'I-' + chunk_type
                    
            elif self.labeling == 'BIOES':
                if chunk_end - chunk_start == 1:
                    tags[chunk_start] = 'S-' + chunk_type
                else:
                    tags[chunk_start] = 'B-' + chunk_type
                    tags[chunk_end-1] = 'E-' + chunk_type
                    for k in range(chunk_start+1, chunk_end-1):
                        tags[k] = 'I-' + chunk_type
            
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
        chunks = []
        
        prev_tag, prev_type = 'O', 'O'
        chunk_start, chunk_end, chunk_types = -1, -1, []
        
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
            # Breaking because of different types, is holding only in case of `append_types` being True. 
            # In such case, the `prev_tag` must be `B` or `I`. 
            # The breaking operation is equivalent to treating `prev_tag` as `E`. 
            if breaking_for_types and this_trans['append_types'] and (this_type != prev_type):
                this_trans = self.trans[('E', this_tag)]
            
            if this_trans['pre_create_entity']:
                chunks.append((self._vote_in_types(chunk_types, breaking_for_types), 
                               chunk_start, chunk_end))
            if this_trans['update_start']:
                chunk_start = k
            if this_trans['update_end']:
                chunk_end = k + 1
            if this_trans['create_types']:
                chunk_types = [this_type]
            if this_trans['append_types']:
                chunk_types.append(this_type)
                
            prev_tag, prev_type = this_trans['leave_tag'], this_type
            
            
        if prev_tag != 'O':
            chunks.append((self._vote_in_types(chunk_types, breaking_for_types), 
                           chunk_start, chunk_end))
            
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
    
        
        
#     def entities2tags(self, raw_text: str, tokens: TokenSequence, entities: list):
#         """
#         Translate entities to tags. 
#         """
#         tags = ['O' for _ in range(len(tokens))]
#         starts, ends = tokens.start, tokens.end
        
#         # Longer entities are of higher priority
#         entities = sorted(entities, key=lambda ent: len(ent), reverse=True)
#         errors = []
#         mismatches = []
#         for ent in entities:
#             # NOTE: ent_start / ent_end may be modified temporarily
#             ent_text, ent_type = ent['entity'], ent['type']
#             ent_start, ent_end = ent['start'], ent['start']
            
#             # Error data
#             if ent_text != raw_text[ent_start:ent_end]:
#                 if re.sub('[-–]', '-', ent_text) == re.sub('[-–]', '-', raw_text[ent_start:ent_end]):
#                     pass
#                 elif re.sub('\s', ' ', ent_text) == re.sub('\s', ' ', raw_text[ent_start:ent_end]):
#                     pass
#                 elif ent_text == raw_text[ent_start:ent_start+len(ent_text)]:
#                     ent_end = ent_start + len(ent_text)
#                 else:
#                     errors.append([ent_text, raw_text[ent_start:ent_end]])
#                     continue
            
#             find_start, idx_start = find_ascending(starts, ent_start)
#             find_end, idx_end = find_ascending(ends, ent_end)
            
#             # Mis-matched data
#             if not find_start:
#                 mismatches.append([raw_text[starts[idx_start-1]:ends[idx_start-1]], 
#                                    raw_text[starts[idx_start-1]:ent_start], 
#                                    raw_text[ent_start:ends[idx_start-1]]])
#             if not find_end:
#                 mismatches.append([raw_text[starts[idx_end]:ends[idx_end]], 
#                                    raw_text[starts[idx_end]:ent_end], 
#                                    raw_text[ent_end:ends[idx_end]]])
                
#             do_labeling = (find_start and find_end)
            
#             # Exceptions: If it is almost found... 
#             if (not find_start) and find_end and (ent_start - starts[idx_start-1] <= 2):
#                 # ent_start = starts[idx_start-1]
#                 idx_start = idx_start - 1
#                 do_labeling = True
#             if find_start and (not find_end) and (ends[idx_end] - ent_end <= 2):
#                 # ent_end = ends[idx_end]
#                 do_labeling = True
            
#             # Check if the target slice is unlabeled...
#             if not all([tags[k] == 'O' for k in range(idx_start, idx_end+1)]):
#                 do_labeling = False
            
#             if do_labeling and self.labeling == 'BIO':
#                 tags[idx_start] = 'B-' + ent_type
#                 for k in range(idx_start+1, idx_end+1):
#                     tags[k] = 'I-' + ent_type
                    
#             if do_labeling and self.labeling == 'BIOES':
#                 if idx_start==idx_end:
#                     tags[idx_start] = 'S-' + ent_type
#                 else:
#                     tags[idx_start] = 'B-' + ent_type
#                     tags[idx_end] = 'E-' + ent_type
#                     for k in range(idx_start+1, idx_end):
#                         tags[k] = 'I-' + ent_type
                
#         return tags, errors, mismatches
    
    
# def tags2entities(raw_text: str, tokens: TokenSequence, tags: list, labeling='BIOES', **kwargs):
#     """
#     Translate tags to entities. 
    
#     Parameters
#     ----------
#     raw_text : str
#         DESCRIPTION.
#     tokens : `TokenSequence`
#         DESCRIPTION.
#     tags : list of str
#         DESCRIPTION.
#     labeling : TYPE, optional
#         DESCRIPTION. The default is 'BIOES'.
#     **kwargs : TYPE
#         Passed to `tags2slices_and_types`.

#     Returns
#     -------
#     entities : list of dict. 
#         [{'entity': ...,
#           'type': ...,
#           'start': ...,
#           'end': ...}, ...]

#     """
#     starts, ends = tokens.start, tokens.end
#     slices_and_types = tags2slices_and_types(tags, labeling=labeling, **kwargs)
    
#     entities = []
#     for ent_slice, ent_type in slices_and_types:
#         ent_start = starts[ent_slice.start]
#         ent_end = ends[ent_slice.stop-1]
#         entity = {'entity': raw_text[ent_start:ent_end], 
#                   'type': ent_type, 
#                   'start': ent_start, 
#                   'end': ent_end}
#         entities.append(entity)
#     return entities



