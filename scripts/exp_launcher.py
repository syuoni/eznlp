# -*- coding: utf-8 -*-
import itertools
import argparse
import time
import subprocess
import multiprocessing
import logging

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
    parser.add_argument('--command', type=str, default='fs', 
                        help="sub-commands")
    parser.add_argument('--num_workers', type=int ,default=0, 
                        help="number of processes to run")
    args = parser.parse_args()
    args.language = dataset2language[args.dataset]
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = ["python", 
               f"scripts/{args.task}.py", 
               f"--dataset {args.dataset}", 
               f"--seed {args.seed}"]
    
    if args.task == 'text_classification':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 50"], 
                       ["--optimizer SGD --lr 0.05", 
                        "--optimizer SGD --lr 0.1", 
                        "--optimizer SGD --lr 0.2", 
                        "--optimizer SGD --lr 0.5", 
                        "--optimizer AdamW --lr 5e-4",
                        "--optimizer AdamW --lr 1e-3",
                        "--optimizer AdamW --lr 2e-3", 
                        "--optimizer Adadelta --lr 0.5",
                        "--optimizer Adadelta --lr 1.0",
                        "--optimizer Adadelta --lr 2.0"], 
                       ["--batch_size 64"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       ["--agg_mode max_pooling", "--agg_mode multiplicative_attention"], 
                       ["fs"]]
        else:
            options = [["--num_epochs 10"], 
                       ["--optimizer AdamW --lr 5e-4 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 5e-4 --finetune_lr 2e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 2e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 2e-5"], 
                       ["--batch_size 32"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm2"], 
                       ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", 
                        "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
                        "--bert_arch ERNIE"]]
    
    elif args.task == 'entity_recognition' and args.language.lower() == 'english':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer SGD --lr 0.1 --batch_size 32"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       # ["--grad_clip -1", "--grad_clip 5"], 
                       # ["", "--use_locked_drop"], 
                       ["--ck_decoder sequence_tagging"],
                       ["fs"], 
                       ["", "--use_elmo"], 
                       ["", "--use_flair"], 
                       ["--char_arch LSTM", "--char_arch Conv"]]
            # options = [["--num_epochs 100"], 
            #            ["--optimizer Adadelta --lr 1.0 --batch_size 64"], 
            #            ["--num_layers 1", "--num_layers 2"], 
            #            ["--ck_decoder span_classification"],
            #            ["--num_neg_chunks 200", "--num_neg_chunks 100"], 
            #            ["--max_span_size 10", "--max_span_size 5"], 
            #            ["--ck_size_emb_dim 25", "--ck_size_emb_dim 10"], 
            #            ["fs"], 
            #            ["--char_arch LSTM", "--char_arch Conv"]]
            # options = [["--num_epochs 100"], 
            #            ["--optimizer AdamW --lr 1e-3 --batch_size 64", 
            #             "--optimizer Adadelta --lr 1.0 --batch_size 64"], 
            #            ["--num_layers 1", "--num_layers 2"], 
            #            ["--ck_decoder boundary_selection"],
            #            ["--affine_arch FFN", "--affine_arch LSTM"], 
            #            ["--fl_gamma 0.0", "--fl_gamma 1.0", "--fl_gamma 2.0", "--fl_gamma 3.0"],
            #            ["fs"], 
            #            ["--char_arch LSTM", "--char_arch Conv"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 2e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 2e-5"], 
                       ["--batch_size 48"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["--ck_decoder sequence_tagging"],
                       # ["", "--use_locked_drop"], 
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm2"], 
                       ["--bert_arch BERT_base", "--bert_arch BERT_large", 
                        "--bert_arch RoBERTa_base", "--bert_arch RoBERTa_large", 
                        "--bert_arch ALBERT_base", "--bert_arch ALBERT_large", "--bert_arch ALBERT_xlarge", "--bert_arch ALBERT_xxlarge", 
                        "--bert_arch BERT_large_wwm", 
                        "--bert_arch SpanBERT_base", "--bert_arch SpanBERT_large"]]
            # options = [["--num_epochs 50"], 
            #            ["--optimizer AdamW --lr 1e-3 --finetune_lr 5e-5", 
            #             "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-4", 
            #             "--optimizer AdamW --lr 2e-3 --finetune_lr 5e-5", 
            #             "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-4"], 
            #            ["--batch_size 48"], 
            #            ["--scheduler LinearDecayWithWarmup"], 
            #            ["--ck_decoder span_classification"],
            #            ["ft"], 
            #            ["--bert_drop_rate 0.2"], 
            #            ["", "--use_interm2"], 
            #            ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", , 
            #             "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
            #             "--bert_arch ERNIE"]]
            # options = [["--num_epochs 50"], 
            #            ["--optimizer AdamW --lr 1e-3 --finetune_lr 5e-5", 
            #             "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-4", 
            #             "--optimizer AdamW --lr 2e-3 --finetune_lr 5e-5", 
            #             "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-4"], 
            #            ["--batch_size 48"], 
            #            ["--scheduler LinearDecayWithWarmup"], 
            #            ["--ck_decoder boundary_selection"],
            #            ["--affine_arch FFN", "--affine_arch LSTM"], 
            #            ["--fl_gamma 0.0", "--fl_gamma 1.0", "--fl_gamma 2.0", "--fl_gamma 3.0"],
            #            ["ft"], 
            #            ["--bert_drop_rate 0.2"], 
            #            ["", "--use_interm2"], 
            #            ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", 
            #             "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
            #             "--bert_arch ERNIE"]]
            
    elif args.task == 'entity_recognition' and args.language.lower() == 'chinese':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer SGD --lr 0.05", 
                        "--optimizer SGD --lr 0.1", 
                        "--optimizer SGD --lr 0.2", 
                        "--optimizer SGD --lr 0.5", 
                        "--optimizer AdamW --lr 5e-4", 
                        "--optimizer AdamW --lr 1e-3", 
                        "--optimizer AdamW --lr 2e-3", 
                        "--optimizer Adamax --lr 1e-3",
                        "--optimizer Adamax --lr 2e-3",
                        "--optimizer Adamax --lr 5e-3"], 
                       ["--batch_size 32"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       ["--ck_decoder sequence_tagging"],
                       ["fs"], 
                       ["--emb_dim 50"], 
                       # ["", "--use_softword"], 
                       ["", "--use_bigram", "--use_softlexicon"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 5e-4 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 5e-4 --finetune_lr 2e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 2e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 2e-5"], 
                       ["--batch_size 48"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["--ck_decoder sequence_tagging"],
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm2"], 
                       ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", 
                        "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
                        "--bert_arch ERNIE"]]
    
    elif args.task == 'relation_extraction':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer Adadelta --lr 1.0", 
                        "--optimizer AdamW --lr 1e-3", 
                        "--optimizer SGD --lr 0.1"], 
                       ["--batch_size 64"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       ["--num_neg_relations 200", "--num_neg_relations 100"], 
                       ["--ck_size_emb_dim 25", "--ck_size_emb_dim 10"], 
                       ["--ck_label_emb_dim 25", "--ck_label_emb_dim 10"], 
                       ["fs"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 1e-3 --finetune_lr 5e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-4", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 5e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-4"], 
                       ["--batch_size 48"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm2"], 
                       ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", 
                        "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
                        "--bert_arch ERNIE"]]
    
    elif args.task == 'joint_er_re':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer Adadelta --lr 1.0", 
                        "--optimizer AdamW --lr 1e-3", 
                        "--optimizer SGD --lr 0.1"], 
                       ["--batch_size 64"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       ["--ck_decoder span_classification"],
                       ["--num_neg_chunks 200", "--num_neg_chunks 100"], 
                       ["--num_neg_relations 200", "--num_neg_relations 100"], 
                       ["--max_span_size 10", "--max_span_size 5"], 
                       ["--ck_size_emb_dim 10"], 
                       ["--ck_label_emb_dim 10"], 
                       ["fs"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 1e-3 --finetune_lr 5e-5", 
                        "--optimizer AdamW --lr 1e-3 --finetune_lr 1e-4", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 5e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-4"], 
                       ["--batch_size 48"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["--ck_decoder span_classification"],
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm2"], 
                       ["--bert_arch BERT_base", "--bert_arch RoBERTa_base", 
                        "--bert_arch MacBERT_base", "--bert_arch MacBERT_large",  
                        "--bert_arch ERNIE"]]
    
    
    if args.num_workers > 0:
        options.insert(0, ["--no_log_terminal"])
        
    commands = []
    for curr_option in itertools.product(*options):
        curr_option = [x for x in curr_option if len(x) > 0]
        curr_command = " ".join([*COMMAND, *curr_option])
        commands.append(curr_command)
    logger.warning(f"There are {len(commands)} scripts to run...")
    
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
        