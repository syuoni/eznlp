# -*- coding: utf-8 -*-
import os
import re
import pickle
from collections import Counter
import pandas as pd


class Transitions(object):
    def __init__(self):
        self._load_transitions(labeling='BIO')
        self._load_transitions(labeling='BIOES')

    def _load_transitions(self, labeling='BIOES'):
        dirname = os.path.dirname(__file__)
        pkl_fn = f"{dirname}/transitions/transitions-{labeling}.pkl"
        if not os.path.exists(pkl_fn):
            trans = pd.read_excel(f"{dirname}/transitions/transitions.xlsx", labeling, index_col=[0, 1])
            trans = {tr: trans.loc[tr].to_dict() for tr in trans.index.tolist()}
            with open(pkl_fn, 'wb') as f:
                pickle.dump(trans, f)
                
        with open(pkl_fn, 'rb') as f:
            trans = pickle.load(f)
        setattr(self, labeling, trans)

_transitions = Transitions()


def find_ascending(seq, value, start=None, end=None):
    """
    Return (find, idx)
    seq[idx] == value if find being True. 
    seq[idx-1] < value < seq[idx] if find being False. 
    
    If invoke `seq.insert(idx, value)`, then the `seq` is still a ascending list. 
    """
    start = 0 if start is None else start
    end = len(seq) if end is None else end
    if start >= end:
        return None, None
    elif start + 1 == end:
        if seq[start] == value:
            return True, start
        elif seq[start] < value:
            return False, start+1
        else:
            return False, start
    
    mid = (start + end) // 2
    if seq[mid] <= value:
        return find_ascending(seq, value, start=mid, end=end)
    else:
        return find_ascending(seq, value, start=start, end=mid)
    

def entities2tags(raw_text, tokens, entities, labeling='BIOES'):
    tags = ['O' for _ in range(len(tokens))]
    starts, ends = tokens.start, tokens.end
    
    # Longer entities are of higher priority
    entities = sorted(entities, key=lambda ent: ent['end']-ent['start'], reverse=True)
    errors = []
    mismatches = []
    for ent in entities:
        ent_text, ent_type, ent_start, ent_end = ent['entity'], ent['type'], ent['start'], ent['end']
        
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


def _create_entity(ent_text, ent_type, ent_start, ent_end):
    entity = {'entity': ent_text, 
              'type': ent_type, 
              'start': ent_start, 
              'end': ent_end}
    return entity


def _vote_type(ent_types):
    return Counter(ent_types).most_common(1)[0][0]


def check_tags_legal(tags, labeling='BIOES'):
    trans = getattr(_transitions, labeling)
    
    padded_tags = ['O'] + [tag.split('-', maxsplit=1)[0] for tag in tags] + ['O']
    return all([trans[(prev_tag, this_tag)]['legal'] for prev_tag, this_tag in zip(padded_tags[:-1], padded_tags[1:])])


def tags2simple_entities(tags, labeling='BIOES'):
    trans = getattr(_transitions, labeling)
    
    simple_entities = []
    prev_tag = 'O'
    ent_start_idx, ent_stop_idx, ent_types = -1, -1, []
    for idx, tag in enumerate(tags):
        if tag in ('O', '<pad>'):
            this_tag = 'O'
        else:
            if '-' in tag:
                this_tag, this_ent_type = tag.split('-', maxsplit=1)
            else:
                this_tag, this_ent_type = tag, '<pseudo-entity>'
        this_trans = trans[(prev_tag, this_tag)]
        if this_trans['pre_create_entity']:
            simple_entities.append({'type': _vote_type(ent_types), 
                                    'start': ent_start_idx, 
                                    'stop': ent_stop_idx})
        if this_trans['update_start']:
            ent_start_idx = idx
        if this_trans['update_end']:
            ent_stop_idx = idx + 1
        if this_trans['create_types']:
            ent_types = [this_ent_type, this_ent_type]
        if this_trans['append_types']:
            ent_types.append(this_ent_type)
        prev_tag = this_trans['leave_tag']
        
    if prev_tag != 'O':
        simple_entities.append({'type': _vote_type(ent_types), 
                                'start': ent_start_idx, 
                                'stop': ent_stop_idx})
        
    return simple_entities


def tags2entities(raw_text, tokens, tags, labeling='BIOES'):
    starts, ends = tokens.start, tokens.end
    simple_entities = tags2simple_entities(tags, labeling=labeling)
    
    entities = []
    for sim_entity in simple_entities:
        ent_start = starts[sim_entity['start']]
        ent_end = ends[sim_entity['stop']-1]
        entity = _create_entity(raw_text[ent_start:ent_end], 
                                sim_entity['type'], ent_start, ent_end)
        entities.append(entity)
    return entities


def tags2entities_legacy(raw_text, tokens, tags, labeling='BIOES'):
    trans = getattr(_transitions, labeling)
    starts, ends = tokens.start, tokens.end
    
    entities = []
    prev_tag = 'O'
    ent_start, ent_end, ent_types = -1, -1, []
    for start, end, tag in zip(starts, ends, tags):
        if tag == 'O':
            this_tag = 'O'
        else:
            this_tag, this_ent_type = tag.split('-', maxsplit=1)
        this_trans = trans[(prev_tag, this_tag)]
        if this_trans['pre_create_entity']:
            entity = _create_entity(raw_text[ent_start:ent_end], 
                                    _vote_type(ent_types), ent_start, ent_end)
            entities.append(entity)
        if this_trans['update_start']:
            ent_start = start
        if this_trans['update_end']:
            ent_end = end
        if this_trans['create_types']:
            ent_types = [this_ent_type, this_ent_type]
        if this_trans['append_types']:
            ent_types.append(this_ent_type)
        prev_tag = this_trans['leave_tag']
        
    if prev_tag != 'O':
        entity = _create_entity(raw_text[ent_start:ent_end], 
                                _vote_type(ent_types), ent_start, ent_end)
        entities.append(entity)
        
    return entities


def f1_score(data_gold, data_pred, return_raw=False):
    f1_data = []
    for ex_true, ex_pred in zip(data_gold, data_pred):
        entities_true = ex_true['entities']
        entities_pred = ex_pred['entities']
        f1_data.append([sum([ent in entities_pred for ent in entities_true]), 
                        len(entities_true), 
                        len(entities_pred)])
        
    f1_df = pd.DataFrame(f1_data, columns=['TP', 'TP_FN', 'TP_FP'])
    f1_df['precision'] = f1_df['TP'] / f1_df['TP_FP']
    f1_df['recall'] = f1_df['TP'] / f1_df['TP_FN']
    f1_df['f1'] = 2 / (1/f1_df['precision'] + 1/f1_df['recall'])
    micro_f1 = f1_df['f1'].mean()
    
    macro_precision = f1_df['TP'].sum() / f1_df['TP_FP'].sum()
    macro_recall = f1_df['TP'].sum() / f1_df['TP_FN'].sum()
    macro_f1 = 2 / (1/macro_precision + 1/macro_recall)
    if not return_raw:
        return micro_f1, macro_f1
    else:
        return micro_f1, macro_f1, f1_df
    
    
def print_tokens_with_tags(tokens, *tags):
    for tok, *tag in zip(tokens, *tags):
        print('%s\t%s' % (tok, '\t'.join(tag)))
        
        
    