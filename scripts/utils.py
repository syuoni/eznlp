# -*- coding: utf-8 -*-
import os
import argparse
import logging
import re
import json
import spacy
import jieba
import random
import time
import numpy
import sklearn.model_selection
import torch
import allennlp.modules
import transformers
import flair

from eznlp.token import Full2Half
from eznlp.io import TabularIO, CategoryFolderIO, ConllIO, JsonIO, TextClsIO, KarpathyIO, BratIO, Src2TrgIO
from eznlp.io import PostIO
from eznlp.vectors import Vectors, GloVe
from eznlp.training import Trainer, LRLambda, collect_params, check_param_groups
from eznlp.metrics import precision_recall_f1_report

logger = logging.getLogger(__name__)


def add_base_arguments(parser: argparse.ArgumentParser):
    group_debug = parser.add_argument_group('debug')
    group_debug.add_argument('--pdb', default=False, action='store_true', 
                             help="whether to use pdb for debug")
    group_debug.add_argument('--profile', default=False, action='store_true', 
                             help="whether to profile")
    group_debug.add_argument('--no_log_terminal', dest='log_terminal', default=True, action='store_false', 
                             help="whether log to terminal")
    
    group_train = parser.add_argument_group('training hyper-parameters')
    group_train.add_argument('--seed', type=int, default=515, 
                             help="random seed")
    group_train.add_argument('--use_amp', default=False, action='store_true', 
                             help="whether to use amp")
    group_train.add_argument('--train_with_dev', default=False, action='store_true', 
                             help="whether to train with development set")
    group_train.add_argument('--num_epochs', type=int, default=100, 
                             help="number of epochs")
    group_train.add_argument('--batch_size', type=int, default=64, 
                             help="batch size")
    group_train.add_argument('--grad_clip', type=float, default=5.0, 
                             help="gradient clip (negative values are set to `None`)")
    
    group_train.add_argument('--optimizer', type=str, default='AdamW', 
                             help="optimizer", choices=['AdamW', 'SGD', 'Adadelta', 'Adamax'])
    group_train.add_argument('--lr', type=float, default=0.001, 
                             help="learning rate")
    group_train.add_argument('--finetune_lr', type=float, default=2e-5, 
                             help="learning rate for finetuning")
    group_train.add_argument('--scheduler', type=str, default='None', 
                             help='scheduler', choices=['None', 'ReduceLROnPlateau', 'LinearDecayWithWarmup', 'PowerDecayWithWarmup'])
    group_train.add_argument('--num_grad_acc_steps', type=int, default=1, 
                             help="number of gradient accumulation steps")
    
    group_model = parser.add_argument_group('model configurations')
    group_model.add_argument('--emb_dim', type=int, default=100, 
                             help="embedding dim (`0` for w/o embeddings)")
    group_model.add_argument('--emb_freeze', default=False, action='store_true', 
                             help="whether to freeze embedding weights")
    group_model.add_argument('--char_arch', type=str, default='None', choices=['None', 'LSTM', 'GRU', 'Conv'], 
                             help="character-level encoder architecture (None for w/o character-level encoder)")
    group_model.add_argument('--use_bigram', default=False, action='store_true', 
                             help="whether to use bigram")
    group_model.add_argument('--use_softword', default=False, action='store_true', 
                             help="whether to use softword")
    group_model.add_argument('--use_softlexicon', default=False, action='store_true', 
                             help="whether to use softlexicon")
    
    group_model.add_argument('--enc_arch', type=str, default='LSTM', choices=['FFN', 'LSTM', 'GRU', 'Conv', 'Gehring', 'Transformer'], 
                             help="token-level encoder architecture")
    group_model.add_argument('--hid_dim', type=int, default=200, 
                             help="hidden dim")
    group_model.add_argument('--num_layers', type=int, default=1, 
                             help="number of encoder layers")
    group_model.add_argument('--drop_rate', type=float, default=0.5, 
                             help="dropout rate")
    group_model.add_argument('--use_locked_drop', default=False, action='store_true', 
                             help="whether to use locked dropout")
    group_model.add_argument('--use_interm1', default=False, action='store_true', 
                             help="whether to use intermediate1")
    
    group_model.add_argument('--use_elmo', default=False, action='store_true', 
                             help="whether to use ELMo")
    group_model.add_argument('--use_flair', default=False, action='store_true', 
                             help="whether to use Flair")
    group_model.add_argument('--bert_arch', type=str, default='None', 
                             help="bert-like architecture (None for w/o bert-like)")
    group_model.add_argument('--bert_cased', default=False, action='store_true', 
                             help="whether to use the cased (case-sensitive) version") 
    group_model.add_argument('--bert_max_length', type=int, default=512, 
                             help="maximum length of subtokenized tokens")
    group_model.add_argument('--bert_drop_rate', type=float, default=0.2, 
                             help="dropout rate for BERT")
    group_model.add_argument('--bert_freeze', default=False, action='store_true', 
                             help="whether to freeze BERT weights")
    group_model.add_argument('--bert_reinit', default=False, action='store_true', 
                             help="whether to reinit BERT weights")
    group_model.add_argument('--use_interm2', default=False, action='store_true', 
                             help="whether to use intermediate2")
    return parser


def parse_to_args(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    args.grad_clip = None if args.grad_clip < 0 else args.grad_clip
    
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    return args



spacy_nlp_en = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])
spacy_nlp_de = spacy.load("de_core_news_sm", disable=['tagger', 'parser', 'ner'])


dataset2language = {'conll2003': 'English', 
                    'conll2003nff': 'English', 
                    'conll2012': 'English', 
                    'ace2004': 'English', 
                    'ace2005': 'English', 
                    'genia': 'English', 
                    'genia_yu2020acl': 'English', 
                    'kbp2017': 'English', 
                    'nne': 'English', 
                    'conll2004': 'English', 
                    'SciERC': 'English', 
                    'SciERC-eberts2020ecai': 'English', 
                    'ace2005_rel': 'English',
                    'ResumeNER': 'Chinese', 
                    'WeiboNER': 'Chinese', 
                    'SIGHAN2006': 'Chinese', 
                    'conll2012_zh': 'Chinese', 
                    'ontonotesv4_zh': 'Chinese',
                    'yidu_s4k': 'Chinese', 
                    'cmeee': 'Chinese', 
                    'cmeie': 'Chinese', 
                    'chip_ctc': 'Chinese', 
                    'chip_sts': 'Chinese', 
                    'kuake_qic': 'Chinese', 
                    'kuake_qtr': 'Chinese', 
                    'kuake_qqr': 'Chinese', 
                    'CLERD': 'Chinese', 
                    'yelp2013': 'English', 
                    'imdb': 'English', 
                    'yelp_full': 'English', 
                    'ChnSentiCorp': 'Chinese', 
                    'THUCNews_10': 'Chinese', 
                    'multi30k': ('English', 'German'), 
                    'iwslt14': ('English', 'German'), 
                    'flickr8k': 'English', 
                    'flickr30k': 'English', 
                    'mscoco': 'English'}
dataset2language.update({f'ADE_cv{k}': 'English' for k in range(10)})
dataset2language.update({f'ace2004_rel_cv{k}': 'English' for k in range(5)})
dataset2language.update({f'HwaMei_{s}': 'Chinese' for s in range(500, 1201, 100)})
dataset2language.update({'HwaMei_Privacy': 'Chinese', 'HwaMei_Privacy_Shaoyang': 'Chinese'})

def load_data(args: argparse.Namespace):
    if args.dataset.startswith('conll2003'):
        io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', document_sep_starts=["-DOCSTART-"], case_mode='None', number_mode='Zeros')
        train_data = io.read("data/conll2003/eng.train")
        dev_data   = io.read("data/conll2003/eng.testa")
        test_data  = io.read("data/conll2003/eng.testb")
        
        if 'nff' in args.dataset:
            json_io = JsonIO(text_key='tokens', 
                             chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', retain_keys=['doc_idx'], 
                             case_mode='None', number_mode='Zeros')
            test_data = json_io.read("data/conll2003/eng.testb.nested.gold.json")
        
        if args.corrupt_rate > 0 and args.doc_level:
            set_chunks_gold = [ex['chunks'] for ex in train_data]
            
            json_io = JsonIO(is_tokenized=True, case_mode='None', number_mode='Zeros')
            train_data = json_io.read(f"data/conll2003/eng.train.corrupted({args.corrupt_rate:.1f}, 1).json")
            # train_data = json_io.read(f"data/conll2003/eng.train.sys.corrupted({args.corrupt_rate:.1f}, 1).json")
            set_chunks_corr = [ex['chunks'] for ex in train_data]
            scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_corr)
            logger.warning(f"Loading data with corruption rate of {args.corrupt_rate:.1f} \n"
                           f"Corruption Retrieval F1-score: {ave_scores['micro']['f1']*100:2.3f}%")
        
        
    elif args.dataset == 'conll2012':
        io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', sentence_sep_starts=["#end", "pt/"], document_sep_starts=["#begin"], encoding='utf-8', case_mode='None', number_mode='Zeros')
        train_data = io.read("data/conll2012/train.english.v4_gold_conll")
        dev_data   = io.read("data/conll2012/dev.english.v4_gold_conll")
        test_data  = io.read("data/conll2012/test.english.v4_gold_conll")
        
    elif args.dataset == 'ace2004':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    case_mode='None', number_mode='Zeros')
        train_data = io.read("data/ace-lu2015emnlp/ACE2004/train.json")
        dev_data   = io.read("data/ace-lu2015emnlp/ACE2004/dev.json")
        test_data  = io.read("data/ace-lu2015emnlp/ACE2004/test.json")
        
    elif args.dataset == 'ace2005':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    case_mode='None', number_mode='Zeros')
        train_data = io.read("data/ace-lu2015emnlp/ACE2005/train.json")
        dev_data   = io.read("data/ace-lu2015emnlp/ACE2005/dev.json")
        test_data  = io.read("data/ace-lu2015emnlp/ACE2005/test.json")
        
        if args.corrupt_rate > 0:
            set_chunks_gold = [ex['chunks'] for ex in train_data]
            train_data = io.read(f"data/ace-lu2015emnlp/ACE2005/train.corrupted({args.corrupt_rate:.1f}, 1).json")
            # train_data = io.read(f"data/ace-lu2015emnlp/ACE2005/train.sys.corrupted({args.corrupt_rate:.1f}, 1).json")
            set_chunks_corr = [ex['chunks'] for ex in train_data]
            scores, ave_scores = precision_recall_f1_report(set_chunks_gold, set_chunks_corr)
            logger.warning(f"Loading data with corruption rate of {args.corrupt_rate:.1f} \n"
                           f"Corruption Retrieval F1-score: {ave_scores['micro']['f1']*100:2.3f}%")
        
    elif args.dataset.startswith('genia'):
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', retain_keys=['doc_key', 'bibliomisc'], 
                    case_mode='None', number_mode='Zeros')
        if args.dataset == 'genia':
            train_data = io.read("data/genia/term.train.json")
            dev_data   = io.read("data/genia/term.dev.json")
            test_data  = io.read("data/genia/term.test.json")
        elif args.dataset == 'genia_yu2020acl':
            train_data = io.read("data/genia-yu2020acl/train_dev.json")
            dev_data   = []
            test_data  = io.read("data/genia-yu2020acl/test.json")
        
    elif args.dataset == 'kbp2017':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', retain_keys=['org_id'], 
                    case_mode='None', number_mode='Zeros')
        train_data = io.read("data/kbp2017-shen2022acl/kbp17_train_context.json")
        dev_data   = io.read("data/kbp2017-shen2022acl/kbp17_dev_context.json")
        test_data  = io.read("data/kbp2017-shen2022acl/kbp17_test_context.json")
        
    elif args.dataset == 'nne':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', retain_keys=['org_id'], 
                    case_mode='None', number_mode='Zeros')
        train_data = io.read("data/nne-shen2022acl/nne_train_context.json")
        dev_data   = io.read("data/nne-shen2022acl/nne_dev_context.json")
        test_data  = io.read("data/nne-shen2022acl/nne_test_context.json")
        
    elif args.dataset == 'conll2004':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    retain_keys=['orig_id'], case_mode='None', number_mode='Zeros')
        train_data = io.read("data/conll2004/conll04_train.json")
        dev_data   = io.read("data/conll2004/conll04_dev.json")
        test_data  = io.read("data/conll2004/conll04_test.json")
        
    elif args.dataset == 'SciERC': 
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    retain_keys=['doc_key'], encoding='utf-8', case_mode='None', number_mode='Zeros')
        train_data = io.read("data/SciERC/train.json")
        dev_data   = io.read("data/SciERC/dev.json")
        test_data  = io.read("data/SciERC/test.json")
        
    elif args.dataset == 'SciERC-eberts2020ecai':
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    retain_keys=['orig_id'], case_mode='None', number_mode='Zeros')
        train_data = io.read("data/SciERC-eberts2020ecai/scierc_train.json")
        dev_data   = io.read("data/SciERC-eberts2020ecai/scierc_dev.json")
        test_data  = io.read("data/SciERC-eberts2020ecai/scierc_test.json")
        
    elif args.dataset.startswith('ADE_cv'):
        io = JsonIO(text_key='tokens', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    case_mode='None', number_mode='Zeros')
        k = int(args.dataset.replace('ADE_cv', ''))
        train_data = io.read(f"data/ADE/ade_split_{k}_train.json")
        dev_data   = []
        test_data  = io.read(f"data/ADE/ade_split_{k}_test.json")
        
    elif args.dataset.startswith('ace2004_rel_cv'):
        io = JsonIO(relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    retain_keys=['doc_key'], case_mode='None', number_mode='Zeros')
        k = int(args.dataset.replace('ace2004_rel_cv', ''))
        train_data = io.read(f"data/ace-luan2019naacl/ace04/cv{k}.train.json")
        dev_data   = []
        test_data  = io.read(f"data/ace-luan2019naacl/ace04/cv{k}.test.json")
        
    elif args.dataset == 'ace2005_rel':
        io = JsonIO(relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    retain_keys=['doc_key'], case_mode='None', number_mode='Zeros')
        train_data = io.read("data/ace-luan2019naacl/ace05/train.json")
        dev_data   = io.read("data/ace-luan2019naacl/ace05/dev.json")
        test_data  = io.read("data/ace-luan2019naacl/ace05/test.json")
        
        set_chunks_false = {866: [('PER', 22, 23), ('PER', 24, 25), ('PER', 28, 29)]}
        for idx, chunks_false in set_chunks_false.items():
            train_data[idx]['chunks'] = [ck for ck in train_data[idx]['chunks'] if not (ck in chunks_false)]
            train_data[idx]['relations'] = [rel for rel in train_data[idx]['relations'] if not (rel[1] in chunks_false or rel[2] in chunks_false)]
        
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
        # https://github.com/v-mipeng/LexiconAugmentedNER/issues/3#issuecomment-634563407
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8', token_sep="", pad_token="")
        train_data = conll_io.read("data/SIGHAN2006/train.txt")
        dev_data   = conll_io.read("data/SIGHAN2006/test.txt")
        test_data  = conll_io.read("data/SIGHAN2006/test.txt")
        
    elif args.dataset == 'conll2012_zh':
        conll_io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', sentence_sep_starts=["#end"], document_sep_starts=["#begin"], encoding='utf-8', token_sep="", pad_token="")
        train_data = conll_io.read("data/conll2012/train.chinese.v4_gold_conll")
        dev_data   = conll_io.read("data/conll2012/dev.chinese.v4_gold_conll")
        test_data  = conll_io.read("data/conll2012/test.chinese.v4_gold_conll")
        train_data = conll_io.flatten_to_characters(train_data)
        dev_data   = conll_io.flatten_to_characters(dev_data)
        test_data  = conll_io.flatten_to_characters(test_data)
        
    elif args.dataset == 'ontonotesv4_zh':
        io = ConllIO(text_col_id=2, tag_col_id=3, scheme='OntoNotes', sentence_sep_starts=["#end"], document_sep_starts=["#begin"], encoding='utf-8', token_sep="", pad_token="")
        # Translate full-width characters to half-width ones, including "！", "？"
        # io = ConllIO(text_col_id=2, tag_col_id=3, scheme='OntoNotes', sentence_sep_starts=["#end"], document_sep_starts=["#begin"], encoding='utf-8', token_sep="", pad_token="", pre_text_normalizer=Full2Half.full2half)
        train_data = io.read("data/ontonotesv4/train.chinese.vz_gold_conll")
        dev_data   = io.read("data/ontonotesv4/dev.chinese.vz_gold_conll")
        test_data  = io.read("data/ontonotesv4/test.chinese.vz_gold_conll")
        train_data = io.flatten_to_characters(train_data)
        dev_data   = io.flatten_to_characters(dev_data)
        test_data  = io.flatten_to_characters(test_data)
        # Che et al. (2013)
        # we selected the four most common named entity types, i.e., 
        # PER (Person), LOC (Location), ORG (Organization) and GPE (Geo-Political Entities), and discarded the others.
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                entry['chunks'] = [ck for ck in entry['chunks'] if ck[0] in ('PERSON', 'LOC', 'ORG', 'GPE')]
        
    elif args.dataset == 'yidu_s4k':
        io = JsonIO(is_tokenized=False, tokenize_callback='char', 
                    text_key='originalText', chunk_key='entities', chunk_type_key='label_type', chunk_start_key='start_pos', chunk_end_key='end_pos', 
                    is_whole_piece=False, encoding='utf-8-sig', token_sep="", pad_token="")
        train_data = io.read("data/yidu_s4k/subtask1_training_part1.txt") + io.read("data/yidu_s4k/subtask1_training_part2.txt")
        # train_data, dev_data = sklearn.model_selection.train_test_split(train_data, test_size=0.2, random_state=args.seed)
        dev_data   = []
        test_data  = io.read("data/yidu_s4k/subtask1_test_set_with_answer.json")
        
        
    elif args.dataset == 'cmeee':
        io = JsonIO(is_tokenized=False, tokenize_callback='char', text_key='text', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start_idx', chunk_end_key='end_idx', 
                    encoding='utf-8', token_sep="", pad_token="")
        train_data = io.read("data/cblue/CMeEE/CMeEE_train_vz.json")
        dev_data   = io.read("data/cblue/CMeEE/CMeEE_dev_vz.json")
        test_data  = io.read("data/cblue/CMeEE/CMeEE_test_vz.json")
        
    elif args.dataset == 'cmeie':
        io = JsonIO(is_tokenized=False, tokenize_callback='char', text_key='text', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    encoding='utf-8', token_sep="", pad_token="")
        train_data = io.read("data/cblue/CMeIE/CMeIE_train_vz.json")
        dev_data   = io.read("data/cblue/CMeIE/CMeIE_dev_vz.json")
        test_data  = io.read("data/cblue/CMeIE/CMeIE_test_vz.json")
        
        
    elif args.dataset == 'CLERD':
        io = BratIO(tokenize_callback='char', has_ins_space=False, parse_attrs=False, parse_relations=True, 
                    max_len=500, line_sep="\n", allow_broken_chunk_text=True, consistency_mapping={'[・;é]': '、'}, 
                    encoding='utf-8', token_sep="", pad_token="")
        train_data = io.read_folder("data/CLERD/relation_extraction/Training")
        dev_data   = io.read_folder("data/CLERD/relation_extraction/Validation")
        test_data  = io.read_folder("data/CLERD/relation_extraction/Testing")
        
        post_io = PostIO(verbose=False)
        kwargs = {'max_span_size': 20, 
                  'chunk_type_mapping': lambda x: x.split('-')[0] if x not in ('Physical', 'Term') else None, 
                  'relation_type_mapping': lambda x: x if x not in ('Coreference', ) else None}
        train_data = post_io.map(train_data, **kwargs)
        dev_data   = post_io.map(dev_data, **kwargs)
        test_data  = post_io.map(test_data, **kwargs)
        
        
    elif (args.dataset.startswith('HwaMei') and 'Privacy' not in args.dataset):
        io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', chunk_text_key=None, 
                    attribute_key='attributes', attribute_type_key='type', attribute_chunk_key='entity', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    is_whole_piece=False, retain_keys=['visit_id', 'split'], encoding='utf-8', token_sep="", pad_token="")
        train_data = io.read("data/HwaMei/v20211230/train.json")
        dev_data   = io.read("data/HwaMei/v20211230/dev.json")
        test_data  = io.read("data/HwaMei/v20211230/test.json")
        
        size = int(args.dataset.split('_')[-1])
        if size > 500:
            with open("data/HwaMei/v20211230/splits.json", 'r', encoding='utf-8') as f:
                splits = json.load(f)
            ext_ids = (splits['reserve'] + splits['iaa'])[:size-500]
            ext_data = io.read("data/HwaMei/v20211230/reserve.json") + io.read("data/HwaMei/v20211230/iaa.json")
            ext_data = [entry for entry in ext_data if entry['visit_id'] in ext_ids]
            train_data += ext_data
        assert len(set([entry['visit_id'] for entry in train_data])) == size - 200
        
    elif (args.dataset.startswith('HwaMei') and 'Privacy' in args.dataset):
        io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', chunk_text_key=None, 
                    is_whole_piece=False, retain_keys=['visit_id', 'split'], encoding='utf-8', token_sep="", pad_token="")
        train_data = io.read("data/HwaMei/v20221201/privacy-train.json")
        dev_data   = io.read("data/HwaMei/v20221201/privacy-dev.json")
        test_data  = io.read("data/HwaMei/v20221201/privacy-test.json")
        
        if 'Shaoyang' in args.dataset:
            test_data = io.read("data/HwaMei/v20230221/privacy-shaoyang-v1.json")
        
        
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
        
    elif args.dataset == 'chip_ctc':
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='text', mapping={" ": ""}, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/cblue/CHIP-CTC/CHIP-CTC_train.json")
        dev_data   = io.read("data/cblue/CHIP-CTC/CHIP-CTC_dev.json")
        test_data  = io.read("data/cblue/CHIP-CTC/CHIP-CTC_test.json")
        
    elif args.dataset == 'chip_sts':
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='text1', paired_text_key='text2', mapping={" ": ""}, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/cblue/CHIP-STS/CHIP-STS_train.json")
        dev_data   = io.read("data/cblue/CHIP-STS/CHIP-STS_dev.json")
        test_data  = io.read("data/cblue/CHIP-STS/CHIP-STS_test.json")
        
    elif args.dataset == 'kuake_qic':
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query', mapping={" ": ""}, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_test.json")
        
    elif args.dataset == 'kuake_qtr':
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query', paired_text_key='title', mapping={" ": ""}, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_test.json")
        
    elif args.dataset == 'kuake_qqr':
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query1', paired_text_key='query2', mapping={" ": ""}, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_test.json")
        
        
    elif args.dataset == 'multi30k':
        io = Src2TrgIO(tokenize_callback=spacy_nlp_de, trg_tokenize_callback=spacy_nlp_en, encoding='utf-8', verbose=args.log_terminal, 
                       case_mode='Lower', number_mode='None')
        train_data = io.read("data/multi30k/train.en", "data/multi30k/train.de")
        dev_data   = io.read("data/multi30k/val.en", "data/multi30k/val.de")
        test_data  = io.read("data/multi30k/test2016.en", "data/multi30k/test2016.de")
        
    elif args.dataset == 'iwslt14':
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        train_data = io.read("data/iwslt14.tokenized.de-en/train.en", "data/iwslt14.tokenized.de-en/train.de")
        dev_data   = io.read("data/iwslt14.tokenized.de-en/valid.en", "data/iwslt14.tokenized.de-en/valid.de")
        test_data  = io.read("data/iwslt14.tokenized.de-en/test.en", "data/iwslt14.tokenized.de-en/test.de")
        
    elif args.dataset == 'flickr8k':
        io = KarpathyIO(img_folder="data/flickr8k/Flicker8k_Dataset", check_img_path=True)
        train_data, dev_data, test_data = io.read("data/flickr8k/flickr8k-karpathy2015cvpr.json")
        
    elif args.dataset == 'flickr30k':
        io = KarpathyIO(img_folder="data/flickr30k/flickr30k-images", check_img_path=True)
        train_data, dev_data, test_data = io.read("data/flickr30k/flickr30k-karpathy2015cvpr.json")
        
    elif args.dataset == 'mscoco':
        io = KarpathyIO(img_folder="data/mscoco/data2014", check_img_path=True)
        train_data, dev_data, test_data = io.read("data/mscoco/mscoco-karpathy2015cvpr.json")
        
    else:
        raise Exception("Dataset does NOT exist", args.dataset)
    
    return train_data, dev_data, test_data



def load_pretrained(pretrained_str, args: argparse.Namespace):
    if pretrained_str.lower() == 'elmo':
        return allennlp.modules.Elmo(options_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_options.json", 
                                     weight_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", 
                                     num_output_representations=1)
        
    elif pretrained_str.lower() == 'flair':
        return (flair.models.LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt"), 
                flair.models.LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt"))
        
    elif args.language.lower() == 'english':
        if pretrained_str.lower().startswith('bert'):
            if 'wwm' in pretrained_str.lower():
                PATH = "assets/transformers/bert-large-cased-whole-word-masking" if args.bert_cased else "assets/transformers/bert-large-uncased-whole-word-masking"
            elif 'base' in pretrained_str.lower():
                PATH = "assets/transformers/bert-base-cased" if args.bert_cased else "assets/transformers/bert-base-uncased"
            elif 'large' in pretrained_str.lower():
                PATH = "assets/transformers/bert-large-cased" if args.bert_cased else "assets/transformers/bert-large-uncased"
            # "cased" means case-sensitive; hence, do_lower_case=False
            # "uncased" means case-unsensitive; hence, do_lower_case=True
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH))
            
        elif pretrained_str.lower().startswith('roberta'):
            if 'base' in pretrained_str.lower():
                PATH = "assets/transformers/roberta-base"
            elif 'large' in pretrained_str.lower():
                PATH = "assets/transformers/roberta-large"
            return (transformers.RobertaModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.RobertaTokenizer.from_pretrained(PATH, add_prefix_space=True))
            
        elif pretrained_str.lower().startswith('deberta'):
            size = re.search("x*(base|large)", pretrained_str.lower())
            if size is not None:
                PATH = f"assets/transformers/microsoft/deberta-{size.group()}"
            return (transformers.DebertaModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.DebertaTokenizer.from_pretrained(PATH, add_prefix_space=True))
            
        elif pretrained_str.lower().startswith('albert'):
            size = re.search("x*(base|large)", pretrained_str.lower())
            if size is not None:
                PATH = f"assets/transformers/albert-{size.group()}-v2"
            # ALBERT by default uses dropout rate of 0
            return (transformers.AlbertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.AlbertTokenizer.from_pretrained(PATH))
            
        elif pretrained_str.lower().startswith('spanbert'):
            if 'base' in pretrained_str.lower():
                PATH = "assets/transformers/SpanBERT/spanbert-base-cased"
            elif 'large' in pretrained_str.lower():
                PATH = "assets/transformers/SpanBERT/spanbert-large-cased"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=False))
            
        elif pretrained_str.lower().startswith('scibert'):
            PATH = "assets/transformers/allenai/scibert_scivocab_cased" if args.bert_cased else "assets/transformers/allenai/scibert_scivocab_uncased"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=(not args.bert_cased)))
            
        elif pretrained_str.lower().startswith('biobert'):
            if 'base' in pretrained_str.lower():
                PATH = "assets/transformers/dmis-lab/biobert-base-cased-v1.1"
            elif 'large' in pretrained_str.lower():
                PATH = "assets/transformers/dmis-lab/biobert-large-cased-v1.1"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=False))
            
        elif pretrained_str.lower().startswith('pubmedbert'):
            if 'full' in pretrained_str.lower():
                PATH = "assets/transformers/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            else:
                PATH = "assets/transformers/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))
            
    elif args.language.lower() == 'chinese':
        if pretrained_str.lower().startswith('bert'):
            if 'wwm' in pretrained_str.lower():
                PATH = "assets/transformers/hfl/chinese-bert-wwm-ext"
            else:
                PATH = "assets/transformers/bert-base-chinese"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))
            
        elif pretrained_str.lower().startswith('roberta'):
            # RoBERTa-like BERT
            # https://github.com/ymcui/Chinese-BERT-wwm#faq
            PATH = "assets/transformers/hfl/chinese-roberta-wwm-ext"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))
            
        elif pretrained_str.lower().startswith('macbert'):
            if 'base' in pretrained_str.lower():
                PATH = "assets/transformers/hfl/chinese-macbert-base"
            elif 'large' in pretrained_str.lower():
                PATH = "assets/transformers/hfl/chinese-macbert-large"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))
            
        elif pretrained_str.lower().startswith('ernie'):
            PATH = "assets/transformers/nghuyong/ernie-1.0"
            return (transformers.AutoModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.AutoTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))
            
        elif pretrained_str.lower().startswith('syuoni_'):
            pretrained_str = pretrained_str.replace('syuoni_', '').replace('_', '-')
            PATH = f"assets/transformers/syuoni/{pretrained_str}"
            return (transformers.BertModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate), 
                    transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True))



def load_vectors(language: str, emb_dim: int, unigram: bool=False, bigram: bool=True):
    if language.lower() == 'english':
        if emb_dim in (50, 100, 200):
            return GloVe(f"assets/vectors/glove.6B.{emb_dim}d.txt")
        elif emb_dim == 300:
            return GloVe("assets/vectors/glove.840B.300d.txt")
    elif language.lower() == 'chinese':
        if unigram and emb_dim == 50:
            return Vectors.load("assets/vectors/gigaword_chn.all.a2b.uni.ite50.vec", encoding='utf-8')
        elif bigram and emb_dim == 50:
            return Vectors.load("assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec", encoding='utf-8')
        else:
            if emb_dim == 50:
                return Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
            elif emb_dim == 200:
                return Vectors.load("assets/vectors/tencent/Tencent_AILab_ChineseEmbedding.txt", encoding='utf-8', skiprows=0)
    return None




def header_format(content: str, sep='=', width=100):
    side_width = max(width - len(content) - 2, 10)
    left_width = side_width // 2
    right_width = side_width - left_width
    return f"{sep*left_width} {content} {sep*right_width}"



def build_trainer(model, device, num_train_batches: int, args: argparse.Namespace):
    param_groups = [{'params': model.pretrained_parameters(), 'lr': args.finetune_lr}]
    param_groups.append({'params': collect_params(model, param_groups), 'lr': args.lr})
    assert check_param_groups(model, param_groups)
    optimizer = getattr(torch.optim, args.optimizer)(param_groups)
    
    schedule_by_step = ('warmup' in args.scheduler.lower())
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif args.scheduler == 'LinearDecayWithWarmup':
        num_warmup_epochs = max(2, args.num_epochs // 5)
        lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=num_train_batches*num_warmup_epochs, 
                                                         num_total_steps=num_train_batches*args.num_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'PowerDecayWithWarmup':
        num_warmup_epochs = max(2, args.num_epochs // 5)
        lr_lambda = LRLambda.power_decay_lr_with_warmup(num_warmup_steps=num_train_batches*num_warmup_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None
    
    return Trainer(model, optimizer=optimizer, scheduler=scheduler, schedule_by_step=schedule_by_step, num_grad_acc_steps=args.num_grad_acc_steps,
                   device=device, grad_clip=args.grad_clip, use_amp=args.use_amp)



def profile(trainer, dataloader):
    # raise "out of memory" error if use_cuda=Ture
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        t0 = time.time()
        prof_loader = [batch for _, batch in zip(range(10), dataloader)]
        logger.info(f"Data loading time: {time.time()-t0:.3f}s")
        t0 = time.time()
        trainer.train_epoch(prof_loader)
        logger.info(f"Model training time: {time.time()-t0:.3f}s")
    
    sort_by = "cuda_time_total" if trainer.device.type.startswith('cuda') else "cpu_time_total"
    prof_table = prof.key_averages().table(sort_by=sort_by, row_limit=10)
    logger.info(f"\n{prof_table}")
    return prof
