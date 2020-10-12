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
    


class Transitions(object):
    def __init__(self):
        self._load_transitions(labeling='BIO')
        self._load_transitions(labeling='BIOES')

    def _load_transitions(self, labeling='BIOES'):
        dirname = os.path.dirname(__file__)
        trans = pd.read_excel(f"{dirname}/transitions.xlsx", labeling, index_col=[0, 1])
        trans = {tr: trans.loc[tr].to_dict() for tr in trans.index.tolist()}
        setattr(self, labeling, trans)

_transitions = Transitions()



def entities2tags(raw_text: str, tokens: TokenSequence, entities: list, labeling='BIOES'):
    """
    Transform entities to tags. 
    
    Parameters
    ----------
    raw_text : str
        DESCRIPTION.
    tokens : `TokenSequence`
        DESCRIPTION.
    entities : list of dict. 
        [{'entity': ...,
          'type': ...,
          'start': ...,
          'end': ...}, ...]
    labeling : TYPE, optional
        DESCRIPTION. The default is 'BIOES'.

    Returns
    -------
    tags : list of str
        DESCRIPTION.
    errors : TYPE
        DESCRIPTION.
    mismatches : TYPE
        DESCRIPTION.
        
    """
    tags = ['O' for _ in range(len(tokens))]
    starts, ends = tokens.start, tokens.end
    
    # Longer entities are of higher priority
    entities = sorted(entities, key=lambda ent: len(ent), reverse=True)
    errors = []
    mismatches = []
    for ent in entities:
        # NOTE: ent_start / ent_end may be modified temporarily
        ent_text, ent_type = ent['entity'], ent['type']
        ent_start, ent_end = ent['start'], ent['start']
        
        # Error data
        if ent_text != raw_text[ent_start:ent_end]:
            if re.sub('[-–]', '-', ent_text) == re.sub('[-–]', '-', raw_text[ent_start:ent_end]):
                pass
            elif re.sub('\s', ' ', ent_text) == re.sub('\s', ' ', raw_text[ent_start:ent_end]):
                pass
            elif ent_text == raw_text[ent_start:ent_start+len(ent_text)]:
                ent_end = ent_start + len(ent_text)
            else:
                errors.append([ent_text, raw_text[ent_start:ent_end]])
                continue
        
        find_start, idx_start = find_ascending(starts, ent_start)
        find_end, idx_end = find_ascending(ends, ent_end)
        
        # Mis-matched data
        if not find_start:
            mismatches.append([raw_text[starts[idx_start-1]:ends[idx_start-1]], 
                               raw_text[starts[idx_start-1]:ent_start], 
                               raw_text[ent_start:ends[idx_start-1]]])
        if not find_end:
            mismatches.append([raw_text[starts[idx_end]:ends[idx_end]], 
                               raw_text[starts[idx_end]:ent_end], 
                               raw_text[ent_end:ends[idx_end]]])
            
        do_labeling = (find_start and find_end)
        
        # Exceptions: If it is almost found... 
        if (not find_start) and find_end and (ent_start - starts[idx_start-1] <= 2):
            # ent_start = starts[idx_start-1]
            idx_start = idx_start - 1
            do_labeling = True
        if find_start and (not find_end) and (ends[idx_end] - ent_end <= 2):
            # ent_end = ends[idx_end]
            do_labeling = True
        
        # Check if the target slice is unlabeled...
        if not all([tags[k] == 'O' for k in range(idx_start, idx_end+1)]):
            do_labeling = False
        
        if do_labeling and labeling == 'BIO':
            tags[idx_start] = 'B-' + ent_type
            for k in range(idx_start+1, idx_end+1):
                tags[k] = 'I-' + ent_type
                
        if do_labeling and labeling == 'BIOES':
            if idx_start==idx_end:
                tags[idx_start] = 'S-' + ent_type
            else:
                tags[idx_start] = 'B-' + ent_type
                tags[idx_end] = 'E-' + ent_type
                for k in range(idx_start+1, idx_end):
                    tags[k] = 'I-' + ent_type
            
    return tags, errors, mismatches




def tags2slices_and_types(tags: list, labeling='BIOES', how_different_types='voting'):
    """
    Transform tags to a list of tuple of (slice, type). 

    Parameters
    ----------
    tags : list of str
        DESCRIPTION.
    how_different_types : TYPE, optional
        'voting' or 'breaking'. The default is 'voting'.
    labeling : TYPE, optional
        DESCRIPTION. The default is 'BIOES'.

    Returns
    -------
    slices_and_types : list of tuple
        [(slice, type), ...]
        
    Refs
    -------
    https://github.com/chakki-works/seqeval/blob/master/seqeval/metrics/sequence_labeling.py
    """
    assert how_different_types.lower() in ('voting', 'breaking')
    trans = getattr(_transitions, labeling)
    
    slices_and_types = []
    prev_tag, prev_type = 'O', 'O'
    ent_start_idx, ent_stop_idx, ent_types = -1, -1, []
    for idx, tag in enumerate(tags):
        if tag in ('O', '<pad>'):
            this_tag, this_type = 'O', 'O'
        else:
            if '-' in tag:
                this_tag, this_type = tag.split('-', maxsplit=1)
            else:
                # Typically cascade-tags without types
                this_tag, this_type = tag, '<pseudo-entity>'
                
        this_trans = trans[(prev_tag, this_tag)]
        
        # Breaking because of different types, holds only in case of `append_types` being True. 
        # In such case, the `prev_tag` must be `B` or `I`. 
        # The breaking operation is equivalent to treating `prev_tag` as `E`. 
        if (how_different_types.lower() == 'breaking') and this_trans['append_types'] and (this_type != prev_type):
            this_trans = trans[('E', this_tag)]
        
        if this_trans['pre_create_entity']:
            slices_and_types.append((slice(ent_start_idx, ent_stop_idx), _vote_type(ent_types)))
        if this_trans['update_start']:
            ent_start_idx = idx
        if this_trans['update_end']:
            ent_stop_idx = idx + 1
        if this_trans['create_types']:
            ent_types = [this_type]
        if this_trans['append_types']:
            ent_types.append(this_type)
            
        prev_tag, prev_type = this_trans['leave_tag'], this_type
        
        
    if prev_tag != 'O':
        slices_and_types.append((slice(ent_start_idx, ent_stop_idx), _vote_type(ent_types)))
        
    return slices_and_types



def _vote_type(ent_types, how_different_types='voting'):
    if how_different_types.lower() == 'breaking':
        # All elements must be the same. 
        return ent_types[0]
    else:
        # Assign the first element with 0.5 higher weight. 
        ent_type_counter = Counter(ent_types)
        ent_type_counter[ent_types[0]] += 0.5
        return ent_type_counter.most_common(1)[0][0]
    
    
def tags2entities(raw_text: str, tokens: TokenSequence, tags: list, labeling='BIOES', **kwargs):
    """
    Transform tags to entities. 
    
    Parameters
    ----------
    raw_text : str
        DESCRIPTION.
    tokens : `TokenSequence`
        DESCRIPTION.
    tags : list of str
        DESCRIPTION.
    labeling : TYPE, optional
        DESCRIPTION. The default is 'BIOES'.
    **kwargs : TYPE
        Passed to `tags2slices_and_types`.

    Returns
    -------
    entities : list of dict. 
        [{'entity': ...,
          'type': ...,
          'start': ...,
          'end': ...}, ...]

    """
    starts, ends = tokens.start, tokens.end
    slices_and_types = tags2slices_and_types(tags, labeling=labeling, **kwargs)
    
    entities = []
    for ent_slice, ent_type in slices_and_types:
        ent_start = starts[ent_slice.start]
        ent_end = ends[ent_slice.stop-1]
        entity = {'entity': raw_text[ent_start:ent_end], 
                  'type': ent_type, 
                  'start': ent_start, 
                  'end': ent_end}
        entities.append(entity)
    return entities


def check_tags_legal(tags, labeling='BIOES'):
    """
    Check if the transitions between tags are legal. 
    
    Parameters
    ----------
    tags : list of str
        DESCRIPTION.
    labeling : TYPE, optional
        DESCRIPTION. The default is 'BIOES'.

    """
    trans = getattr(_transitions, labeling)
    
    padded_tags = ['O'] + [tag.split('-', maxsplit=1)[0] for tag in tags] + ['O']
    return all([trans[(prev_tag, this_tag)]['legal'] for prev_tag, this_tag in zip(padded_tags[:-1], padded_tags[1:])])



