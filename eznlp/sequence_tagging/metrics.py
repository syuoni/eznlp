# -*- coding: utf-8 -*-


def _agg_scores_by_key(scores, key, agg='mean'):
    """
    Args
    ----
    scores: list/dict
        list/dict of {'precision': ..., 'recall': ..., 'f1': ...}
    """
    if isinstance(scores, list):
        sum_value = sum(sub_scores[key] for sub_scores in scores)
    else:
        sum_value = sum(sub_scores[key] for _, sub_scores in scores.items())
    if agg == 'sum':
        return sum_value
    elif agg == 'mean':
        return sum_value / len(scores)
    

def _precision_recall_f1(n_gold, n_pred, n_true_positive, zero_division=0):
    precision = n_true_positive / n_pred if n_pred > 0 else zero_division
    recall    = n_true_positive / n_gold if n_gold > 0 else zero_division
    f1 = 2 / (1/precision + 1/recall) if (precision + recall > 0) else zero_division
    return precision, recall, f1


def _prf_scores_over_samples(chunks_gold_data: list, chunks_pred_data: list, **kwargs):
    scores = []
    for chunks_gold, chunks_pred in zip(chunks_gold_data, chunks_pred_data):
        chunks_gold = set(chunks_gold)
        chunks_pred = set(chunks_pred)
        n_gold = len(chunks_gold)
        n_pred = len(chunks_pred)
        n_true_positive = len(chunks_gold & chunks_pred)
        
        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores.append({'n_gold': n_gold,
                       'n_pred': n_pred,
                       'n_true_positive': n_true_positive,
                       'precision': precision, 
                       'recall': recall, 
                       'f1': f1})
    return scores


def _prf_scores_over_types(chunks_gold_data: list, chunks_pred_data: list, **kwargs):
    TYPE_POS = len(chunks_gold_data[0][0]) - 3
    types_set = {ck[TYPE_POS] for chunks_data in [chunks_gold_data, chunks_pred_data] for chunks in chunks_data for ck in chunks}
    
    scores = {}
    for chunk_type in types_set:
        n_gold, n_pred, n_true_positive = 0, 0, 0
        for chunks_gold, chunks_pred in zip(chunks_gold_data, chunks_pred_data):
            chunks_gold = {ck for ck in chunks_gold if ck[TYPE_POS]==chunk_type}
            chunks_pred = {ck for ck in chunks_pred if ck[TYPE_POS]==chunk_type}
            n_gold += len(chunks_gold)
            n_pred += len(chunks_pred)
            n_true_positive += len(chunks_gold & chunks_pred)
            
        precision, recall, f1 = _precision_recall_f1(n_gold, n_pred, n_true_positive, **kwargs)
        scores[chunk_type] = {'n_gold': n_gold,
                              'n_pred': n_pred,
                              'n_true_positive': n_true_positive,
                              'precision': precision, 
                              'recall': recall, 
                              'f1': f1}
    return scores


def precision_recall_f1_report(chunks_gold_data: list, chunks_pred_data: list, macro_over='types', **kwargs):
    """
    Args
    ----
    chunks_gold/chunks_pred: list
        list of list of (chunk_type, chunk_start, chunk_end), or
        list of list of (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        
    References
    ----------
    https://github.com/chakki-works/seqeval
    """
    if macro_over == 'types':
        scores = _prf_scores_over_types(chunks_gold_data, chunks_pred_data, **kwargs)
    elif macro_over == 'samples':
        scores = _prf_scores_over_samples(chunks_gold_data, chunks_pred_data, **kwargs)
    else:
        raise ValueError(f"Invalid macro_over {macro_over}")
        
        
    ave_scores = {}
    ave_scores['macro'] = {key: _agg_scores_by_key(scores, key, agg='mean') for key in ['precision', 'recall', 'f1']}
    ave_scores['micro'] = {key: _agg_scores_by_key(scores, key, agg='sum') for key in ['n_gold', 'n_pred', 'n_true_positive']}
    
    micro_precision, micro_recall, micro_f1 = _precision_recall_f1(ave_scores['micro']['n_gold'], 
                                                                   ave_scores['micro']['n_pred'], 
                                                                   ave_scores['micro']['n_true_positive'], **kwargs)
    ave_scores['micro'].update({'precision': micro_precision, 
                                'recall': micro_recall, 
                                'f1': micro_f1})
    
    return scores, ave_scores
    

