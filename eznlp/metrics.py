# -*- coding: utf-8 -*-
from typing import List


def _agg_scores_by_key(scores, key, agg_mode='mean'):
    """
    Parameters
    ----------
    scores: list or dict
        list or dict of {'precision': ..., 'recall': ..., 'f1': ...}
    """
    if len(scores) == 0:
        return 0
    
    if isinstance(scores, list):
        sum_value = sum(sub_scores[key] for sub_scores in scores)
    else:
        sum_value = sum(sub_scores[key] for _, sub_scores in scores.items())
    if agg_mode == 'sum':
        return sum_value
    elif agg_mode == 'mean':
        return sum_value / len(scores)


def _precision_recall_f1(n_gold, n_pred, n_true_positive, zero_division=0):
    precision = n_true_positive / n_pred if n_pred > 0 else zero_division
    recall    = n_true_positive / n_gold if n_gold > 0 else zero_division
    f1 = 2 / (1/precision + 1/recall) if (precision + recall > 0) else zero_division
    return precision, recall, f1


def _prf_scores_over_samples(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]], **kwargs):
    scores = []
    for tuples_gold, tuples_pred in zip(list_tuples_gold, list_tuples_pred):
        tuples_gold, tuples_pred = set(tuples_gold), set(tuples_pred)
        n_gold, n_pred = len(tuples_gold), len(tuples_pred)
        n_true_positive = len(tuples_gold & tuples_pred)
        
        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores.append({'n_gold': n_gold,
                       'n_pred': n_pred,
                       'n_true_positive': n_true_positive,
                       'precision': precision, 
                       'recall': recall, 
                       'f1': f1})
    return scores


def _prf_scores_over_types(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]], type_pos=0, **kwargs):
    tuples_set = {tp for list_tuples in [list_tuples_gold, list_tuples_pred] for tuples in list_tuples for tp in tuples}
    if len(tuples_set) == 0:
        return {}
    
    types_set = {tp[type_pos] for tp in tuples_set}
    
    scores = {}
    for _type in types_set:
        n_gold, n_pred, n_true_positive = 0, 0, 0
        for tuples_gold, tuples_pred in zip(list_tuples_gold, list_tuples_pred):
            tuples_gold = {tp for tp in tuples_gold if tp[type_pos]==_type}
            tuples_pred = {tp for tp in tuples_pred if tp[type_pos]==_type}
            n_gold += len(tuples_gold)
            n_pred += len(tuples_pred)
            n_true_positive += len(tuples_gold & tuples_pred)
            
        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores[_type] = {'n_gold': n_gold,
                         'n_pred': n_pred,
                         'n_true_positive': n_true_positive,
                         'precision': precision, 
                         'recall': recall, 
                         'f1': f1}
    return scores


def precision_recall_f1_report(list_tuples_gold: List[List[tuple]], list_tuples_pred: List[List[tuple]], macro_over='types', **kwargs):
    """
    Parameters
    ----------
    list_tuples_{gold, pred}: a list of lists of tuples
        A tuple of chunk or entity is in format of (chunk_type, chunk_start, chunk_end) or
                                                   (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        A tuple of relation is in format of (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end))
        
    macro_over: str
        'types' or 'samples'
        
    type_pos: int
        The position indicating type in a tuple
        
        
    References
    ----------
    https://github.com/chakki-works/seqeval
    """
    assert len(list_tuples_gold) == len(list_tuples_pred)
    
    if macro_over == 'types':
        scores = _prf_scores_over_types(list_tuples_gold, list_tuples_pred, **kwargs)
    elif macro_over == 'samples':
        scores = _prf_scores_over_samples(list_tuples_gold, list_tuples_pred, **kwargs)
    else:
        raise ValueError(f"Invalid `macro_over` {macro_over}")
        
    ave_scores = {}
    ave_scores['macro'] = {key: _agg_scores_by_key(scores, key, agg_mode='mean') for key in ['precision', 'recall', 'f1']}
    ave_scores['micro'] = {key: _agg_scores_by_key(scores, key, agg_mode='sum') for key in ['n_gold', 'n_pred', 'n_true_positive']}
    
    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(ave_scores['micro']['n_gold'], 
                                                                   ave_scores['micro']['n_pred'], 
                                                                   ave_scores['micro']['n_true_positive'], **kwargs)
    ave_scores['micro'].update({'precision': micro_precision, 
                                'recall': micro_recall, 
                                'f1': micro_f1})
    
    return scores, ave_scores

