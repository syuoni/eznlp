# -*- coding: utf-8 -*-
import argparse
import logging
import spacy
import jieba
import sklearn.model_selection
import torch

from eznlp.token import LexiconTokenizer
from eznlp.metrics import precision_recall_f1_report
from eznlp.pretrained import Vectors
from eznlp.io import TabularIO, CategoryFolderIO, ConllIO, JsonIO
from eznlp.training.utils import LRLambda
from eznlp.training.utils import count_params, collect_params, check_param_groups

logger = logging.getLogger(__name__)


spacy_nlp_en = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])


def load_data(args: argparse.Namespace):
    if args.dataset == 'conll2003':
        conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', case_mode='None', number_mode='Zeros')
        train_data = conll_io.read("data/conll2003/eng.train")
        dev_data   = conll_io.read("data/conll2003/eng.testa")
        test_data  = conll_io.read("data/conll2003/eng.testb")
    elif args.dataset == 'conll2012':
        conll_io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', line_sep_starts=["#begin", "#end", "pt/"], encoding='utf-8', case_mode='None', number_mode='Zeros')
        train_data = conll_io.read("data/conll2012/train.english.v4_gold_conll")
        dev_data   = conll_io.read("data/conll2012/dev.english.v4_gold_conll")
        test_data  = conll_io.read("data/conll2012/test.english.v4_gold_conll")
        
    elif args.dataset == 'conll2004':
        json_io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', case_mode='None', number_mode='Zeros')
        train_data = json_io.read("data/conll2004/conll04_train.json")
        dev_data   = json_io.read("data/conll2004/conll04_dev.json")
        test_data  = json_io.read("data/conll2004/conll04_test.json")
        
    elif args.dataset == 'ResumeNER':
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BMES', encoding='utf-8', token_sep="", pad_token="")
        train_data = conll_io.read("data/ResumeNER/train.char.bmes")
        dev_data   = conll_io.read("data/ResumeNER/dev.char.bmes")
        test_data  = conll_io.read("data/ResumeNER/test.char.bmes")
    elif args.dataset == 'WeiboNER':
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8', token_sep="", pad_token="", pre_text_normalizer=lambda x: x[0])
        train_data = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.train")
        dev_data   = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.dev")
        test_data  = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.test")
    elif args.dataset == 'SIGHAN2006':
        # https://github.com/v-mipeng/LexiconAugmentedNER/issues/3
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8', token_sep="", pad_token="")
        train_data = conll_io.read("data/SIGHAN2006/train.txt")
        dev_data   = conll_io.read("data/SIGHAN2006/test.txt")
        test_data  = conll_io.read("data/SIGHAN2006/test.txt")
        if args.command in ('finetune', 'ft'):
            for data in [train_data, dev_data, test_data]:
                for data_entry in data:
                    data_entry['tokens'] = data_entry['tokens'][:510]
                    
    elif args.dataset == 'conll2012_zh':
        conll_io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', line_sep_starts=["#begin", "#end", "pt/"], encoding='utf-8', token_sep="", pad_token="")
        train_data = conll_io.read("data/conll2012/train.chinese.v4_gold_conll")
        dev_data   = conll_io.read("data/conll2012/dev.chinese.v4_gold_conll")
        test_data  = conll_io.read("data/conll2012/test.chinese.v4_gold_conll")
        
    elif args.dataset == 'yelp2013':
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8', verbose=args.log_terminal, 
                               case_mode='Lower', number_mode='None')
        train_data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.train.ss")
        dev_data   = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.dev.ss")
        test_data  = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss")
    elif args.dataset == 'imdb':
        folder_io = CategoryFolderIO(categories=["pos", "neg"], mapping={"<br />": "\n"}, tokenize_callback=spacy_nlp_en, encoding='utf-8', verbose=args.log_terminal, 
                             case_mode='lower', number_mode='None')
        train_data = folder_io.read("data/imdb/train")
        test_data  = folder_io.read("data/imdb/test")
        train_data, dev_data = sklearn.model_selection.train_test_split(train_data, test_size=0.2, random_state=args.seed)
    elif args.dataset == 'yelp_full':
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep=",", mapping={"\\n": "\n", '\\"': '"'}, tokenize_callback=spacy_nlp_en, verbose=args.log_terminal, 
                               case_mode='Lower', number_mode='None')
        train_data = tabular_io.read("data/yelp_review_full/train.csv")
        test_data  = tabular_io.read("data/yelp_review_full/test.csv")
        train_data, dev_data = sklearn.model_selection.train_test_split(train_data, test_size=0.1, random_state=args.seed)
        
    elif args.dataset == 'ChnSentiCorp':
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep='\t', header=0, tokenize_callback=jieba.cut, encoding='utf-8', verbose=args.log_terminal, 
                               case_mode='Lower', number_mode='None')
        train_data = tabular_io.read("data/ChnSentiCorp/train.tsv")
        dev_data   = tabular_io.read("data/ChnSentiCorp/dev.tsv")
        test_data  = tabular_io.read("data/ChnSentiCorp/test.tsv")
    elif args.dataset == 'THUCNews_10':
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep='\t', tokenize_callback=jieba.cut, encoding='utf-8', verbose=args.log_terminal, 
                               case_mode='Lower', number_mode='None')
        train_data = tabular_io.read("data/THUCNews-10/cnews.train.txt")
        dev_data   = tabular_io.read("data/THUCNews-10/cnews.val.txt")
        test_data  = tabular_io.read("data/THUCNews-10/cnews.test.txt")
        
    else:
        raise Exception("Dataset does NOT exist", args.dataset)
        
    if getattr(args, 'use_softword', False) or getattr(args, 'use_softlexicon', False):
        ctb50 = Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
        tokenizer = LexiconTokenizer(ctb50.itos)
        for data in [train_data, dev_data, test_data]:
            for data_entry in data:
                data_entry['tokens'].build_softwords(tokenizer.tokenize)
                data_entry['tokens'].build_softlexicons(tokenizer.tokenize)
                
    if args.dataset in ('ChnSentiCorp', 'THUCNews_10'):
        for data in [train_data, dev_data, test_data]:
            for data_entry in data:
                if len(data_entry['tokens']) > 1200:
                    data_entry['tokens'] = data_entry['tokens'][:300] + data_entry['tokens'][-900:]
                    
    return train_data, dev_data, test_data



def header_format(content: str, sep='=', width=100):
    side_width = max(width - len(content) - 2, 10)
    left_width = side_width // 2
    right_width = side_width - left_width
    return f"{sep*left_width} {content} {sep*right_width}"



def build_trainer(trainer_cls, model, device, num_train_batches: int, args: argparse.Namespace):
    param_groups = [{'params': model.pretrained_parameters(), 'lr': args.finetune_lr}]
    param_groups.append({'params': collect_params(model, param_groups), 'lr': args.lr})
    assert check_param_groups(model, param_groups)
    optimizer = getattr(torch.optim, args.optimizer)(param_groups)
    
    schedule_by_step = False
    if args.scheduler == 'None':
        scheduler = None
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        schedule_by_step = True
        # lr_lambda = LRLambda.constant_lr()
        num_warmup_epochs = max(2, args.num_epochs // 5)
        lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=num_train_batches*num_warmup_epochs, 
                                                         num_total_steps=num_train_batches*args.num_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return trainer_cls(model, optimizer=optimizer, scheduler=scheduler, schedule_by_step=schedule_by_step,
                       device=device, grad_clip=args.grad_clip, use_amp=args.use_amp)
