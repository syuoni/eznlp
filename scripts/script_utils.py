# -*- coding: utf-8 -*-
import argparse
import jieba
import logging

from eznlp.sequence_tagging import precision_recall_f1_report
from eznlp.sequence_tagging.io import ConllIO

logger = logging.getLogger(__name__)


def load_data(args: argparse.Namespace):
    if args.dataset == 'conll2003':
        conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', case_mode='None', number_mode='Zeros')
        train_data = conll_io.read("data/conll2003/eng.train")
        dev_data   = conll_io.read("data/conll2003/eng.testa")
        test_data  = conll_io.read("data/conll2003/eng.testb")
    elif args.dataset == 'ResumeNER':
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BMES', token_sep="", pad_token="")
        train_data = conll_io.read("data/ResumeNER/train.char.bmes", encoding='utf-8')
        dev_data   = conll_io.read("data/ResumeNER/dev.char.bmes", encoding='utf-8')
        test_data  = conll_io.read("data/ResumeNER/test.char.bmes", encoding='utf-8')
    elif args.dataset == 'WeiboNER':
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', token_sep="", pad_token="")
        train_data = conll_io.read("data/WeiboNER/weiboNER.conll.train", encoding='utf-8')
        dev_data   = conll_io.read("data/WeiboNER/weiboNER.conll.dev", encoding='utf-8')
        test_data  = conll_io.read("data/WeiboNER/weiboNER.conll.test", encoding='utf-8')
    elif args.dataset == 'SIGHAN2006':
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', token_sep="", pad_token="")
        train_data = conll_io.read("data/SIGHAN2006/train.txt", encoding='utf-8')
        dev_data   = conll_io.read("data/SIGHAN2006/test.txt", encoding='utf-8')
        test_data  = conll_io.read("data/SIGHAN2006/test.txt", encoding='utf-8')
    else:
        raise Exception("Dataset does NOT exist", args.dataset)
        
    if getattr(args, 'use_softword', False):
        for data in [train_data, dev_data, test_data]:
            for data_entry in data:
                data_entry['tokens'].build_softwords(jieba.tokenize, mode='search')
                
    return train_data, dev_data, test_data



def evaluate_sequence_tagging(trainer, dataset):
    set_chunks_pred = trainer.predict_chunks(dataset)
    set_chunks_gold = [ex['chunks'] for ex in dataset.data]
    
    scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_pred)
    micro_f1, macro_f1 = ave_scores['micro']['f1'], ave_scores['macro']['f1']
    logger.info(f"Micro F1-score: {micro_f1*100:2.3f}%")
    logger.info(f"Macro F1-score: {macro_f1*100:2.3f}%")
    
