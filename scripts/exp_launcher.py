# -*- coding: utf-8 -*-
import argparse
import time
import subprocess
import multiprocessing
import logging
import numpy

from eznlp.training import OptionSampler
from utils import dataset2language


def call_command(command: str):
    logger.warning(f"Starting: {command}")
    subprocess.check_call(command.split())
    logger.warning(f"Ending: {command}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default='entity_recognition', 
                        help="task name")
    parser.add_argument('--dataset', type=str, default='conll2003', 
                        help="dataset name")
    parser.add_argument('--seed', type=int, default=515, 
                        help="random seed")
    parser.add_argument('--use_bert', default=False, action='store_true', 
                        help="whether to use bert-like")
    parser.add_argument('--num_exps', type=int, default=-1, 
                        help="number of experiments to run")
    parser.add_argument('--num_workers', type=int ,default=0, 
                        help="number of processes to run")
    args = parser.parse_args()
    args.language = dataset2language[args.dataset]
    args.num_exps = None if args.num_exps < 0 else args.num_exps
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = f"python scripts/{args.task}.py --dataset {args.dataset} --seed {args.seed}"
    if args.num_workers > 0:
        COMMAND = " ".join([COMMAND, "--no_log_terminal"])
    if args.use_bert:
        COMMAND = " ".join([COMMAND, "@scripts/options/with_bert.opt"])
    else:
        COMMAND = " ".join([COMMAND, "@scripts/options/without_bert.opt"])
    
    if args.task == 'text_classification':
        if not args.use_bert:
            sampler = OptionSampler(num_epochs=50, 
                                    optimizer=['SGD'], lr=[0.05, 0.1, 0.2, 0.5], 
                                    # optimizer=['AdamW'], lr=[5e-4, 1e-3, 2e-3], 
                                    # optimizer=['Adadelta'], lr=[0.5, 1.0, 2.0], 
                                    batch_size=64, 
                                    num_layers=[1, 2], 
                                    agg_mode=['max_pooling', 'multiplicative_attention'])
        else:
            sampler = OptionSampler(num_epochs=10, 
                                    lr=[5e-4, 1e-3, 2e-3], 
                                    finetune_lr=[1e-5, 2e-5], 
                                    batch_size=32, 
                                    bert_drop_rate=0.2, 
                                    use_interm2=[False, True], 
                                    bert_arch=['BERT_base', 'RoBERTa_base'])
        
    elif args.task == 'entity_recognition' and args.language.lower() == 'english':
        if not args.use_bert:
            sampler = OptionSampler(num_epochs=100, 
                                    optimizer=['SGD'], lr=[0.1], 
                                    batch_size=32, 
                                    num_layers=[1, 2], 
                                    # grad_clip=[-1, 5],
                                    # use_locked_drop=[False, True],
                                    ck_decoder='sequence_tagging',
                                    use_elmo=[False, True], 
                                    use_flair=[False, True], 
                                    char_arch=['LSTM', 'Conv'])
            
            # sampler = OptionSampler(num_epochs=100, 
            #                         optimizer=['Adadelta'], lr=[1.0], 
            #                         batch_size=64, 
            #                         num_layers=[1, 2], 
            #                         ck_decoder='span_classification',
            #                         num_neg_chunks=[100, 200], 
            #                         max_span_size=[5, 10], 
            #                         ck_size_emb_dim=[10, 25],
            #                         char_arch=['LSTM', 'Conv'])
            
            # sampler = OptionSampler(num_epochs=100, 
            #                         optimizer=['AdamW'], lr=[1e-3], 
            #                         batch_size=64, 
            #                         num_layers=[1, 2], 
            #                         ck_decoder='boundary_selection',
            #                         affine_arch=['FFN', 'LSTM'],
            #                         sb_epsilon=[0.0, 0.1],
            #                         char_arch=['LSTM', 'Conv'])
        else:
            sampler = OptionSampler(doc_level=True, 
                                    train_with_dev=False, 
                                    num_epochs=50, 
                                    lr=[1e-3, 2e-3], 
                                    # lr=numpy.logspace(-3.1, -2.5, num=100, base=10).tolist(), # 8e-4 ~ 3e-3
                                    finetune_lr=[1e-5, 2e-5], 
                                    # finetune_lr=numpy.logspace(-5.1, -4.5, num=100, base=10).tolist(), # 8e-6 ~ 3e-5
                                    batch_size=48, 
                                    ck_decoder='sequence_tagging',
                                    bert_drop_rate=0.2, 
                                    use_interm2=[False, True], 
                                    bert_arch=['BERT_base', 'RoBERTa_base', 
                                               'BERT_large', 'RoBERTa_large', 
                                               'ALBERT_base', 'ALBERT_large', 'ALBERT_xlarge', 'ALBERT_xxlarge', 
                                               'BERT_large_wwm', 
                                               'SpanBERT_base', 'SpanBERT_large'])
            
            # sampler = OptionSampler(num_epochs=50, 
            #                         lr=[1e-3, 2e-3], 
            #                         finetune_lr=[1e-5, 2e-5], 
            #                         batch_size=48, 
            #                         ck_decoder='span_classification',
            #                         bert_drop_rate=0.2, 
            #                         use_interm2=[False, True], 
            #                         bert_arch=['BERT_base', 'RoBERTa_base'])
            
            # sampler = OptionSampler(num_epochs=50, 
            #                         lr=[1e-3, 2e-3], 
            #                         finetune_lr=[1e-5, 2e-5], 
            #                         batch_size=48, 
            #                         ck_decoder='boundary_selection',
            #                         affine_arch=['FFN', 'LSTM'],
            #                         sb_epsilon=[0.0, 0.1],
            #                         bert_drop_rate=0.2, 
            #                         use_interm2=[False, True], 
            #                         bert_arch=['BERT_base', 'RoBERTa_base'])
            
    elif args.task == 'entity_recognition' and args.language.lower() == 'chinese':
        if not args.use_bert:
            sampler = OptionSampler(num_epochs=100, 
                                    optimizer=['AdamW', 'Adamax'], lr=[5e-4, 1e-3, 2e-3, 5e-3], 
                                    batch_size=32, 
                                    num_layers=[1, 2], 
                                    ck_decoder='sequence_tagging',
                                    # use_bigram=[False, True], 
                                    # use_softword=[False, True], 
                                    use_softlexicon=[False, True])
            
        else:
            sampler = OptionSampler(num_epochs=50, 
                                    lr=[1e-3, 2e-3], 
                                    finetune_lr=[1e-5, 2e-5], 
                                    batch_size=48, 
                                    ck_decoder='sequence_tagging',
                                    bert_drop_rate=0.2, 
                                    use_interm2=[False, True], 
                                    bert_arch=['BERT_base', 'RoBERTa_base', 
                                               'MacBERT_base', 'MacBERT_large', 'ERNIE'])
            
    elif args.task == 'relation_extraction':
        if not args.use_bert:
            sampler = OptionSampler(num_epochs=100, 
                                    # optimizer=['SGD'], lr=[0.1], 
                                    # optimizer=['Adadelta'], lr=[1.0],
                                    optimizer=['AdamW'], lr=[1e-3],
                                    batch_size=64, 
                                    num_layers=[1, 2], 
                                    num_neg_relations=[100, 200], 
                                    ck_size_emb_dim=[10, 25], 
                                    ck_label_emb_dim=[10, 25])
        else:
            sampler = OptionSampler(num_epochs=50, 
                                    lr=[1e-3, 2e-3], 
                                    finetune_lr=[1e-5, 2e-5], 
                                    batch_size=48, 
                                    bert_drop_rate=0.2, 
                                    use_interm2=[False, True], 
                                    bert_arch=['BERT_base', 'RoBERTa_base'])
            
    elif args.task == 'joint_extraction':
        if not args.use_bert:
            sampler = OptionSampler(num_epochs=100, 
                                    # optimizer=['SGD'], lr=[0.1], 
                                    # optimizer=['Adadelta'], lr=[1.0],
                                    optimizer=['AdamW'], lr=[1e-3],
                                    batch_size=64, 
                                    num_layers=[1, 2], 
                                    ck_decoder='span_classification',
                                    num_neg_chunks=[100, 200],
                                    num_neg_relations=[100, 200], 
                                    max_span_size=[5, 10],
                                    ck_size_emb_dim=[10, 25], 
                                    ck_label_emb_dim=[10, 25])
        else:
            sampler = OptionSampler(num_epochs=50, 
                                    lr=[1e-3, 2e-3], 
                                    finetune_lr=[1e-5, 2e-5], 
                                    batch_size=48, 
                                    ck_decoder='span_classification',
                                    bert_drop_rate=0.2, 
                                    use_interm2=[False, True], 
                                    bert_arch=['BERT_base', 'RoBERTa_base'])
    
    options = sampler.sample(args.num_exps)
    commands = [" ".join([COMMAND, *option]) for option in options]
    logger.warning(f"There are {len(commands)} experiments to run...")
    
    if args.num_workers <= 0:
        logger.warning("Starting a single process to run...")
        for curr_command in commands:
            call_command(curr_command)
    else:
        logger.warning(f"Starting {args.num_workers} processes to run...")
        pool = multiprocessing.Pool(processes=args.num_workers)
        for curr_command in commands:
            pool.apply_async(call_command, (curr_command, ))
            # Ensure auto-device allocated before the next process starts...
            time.sleep(60)
        pool.close()
        pool.join()
