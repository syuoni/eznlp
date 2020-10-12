# -*- coding: utf-8 -*-
import pandas as pd


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
        
        
    