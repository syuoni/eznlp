# -*- coding: utf-8 -*-
import itertools
import argparse
import time
import subprocess
import multiprocessing
import logging

"""
python scripts/sequence_tagging_en.py fs
python scripts/sequence_tagging_en.py --batch_size 64 fs
python scripts/sequence_tagging_en.py --optimizer AdamW --lr 1e-3 fs
python scripts/sequence_tagging_en.py --grad_clip -1 fs
python scripts/sequence_tagging_en.py --scheduler ReduceLROnPlateau fs
python scripts/sequence_tagging_en.py --scheduler LinearDecayWithWarmup fs
python scripts/sequence_tagging_en.py --num_layers 1 fs
python scripts/sequence_tagging_en.py --use_locked_drop fs

python scripts/sequence_tagging_en.py --char_arch LSTM fs
python scripts/sequence_tagging_en.py --use_elmo fs
python scripts/sequence_tagging_en.py --use_flair fs

python scripts/sequence_tagging_en.py --scheduler LinearDecayWithWarmup ft
python scripts/sequence_tagging_en.py --scheduler LinearDecayWithWarmup ft --use_roberta
"""


def call_command(command: str):
    logger.warning(f"Starting: {command}")
    subprocess.check_call(command.split())
    logger.warning(f"Ending: {command}")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task', type=str, default='sequence_tagging_en', 
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
    
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = ["python", 
               f"scripts/{args.task}.py", 
               f"--dataset {args.dataset}", 
               f"--seed {args.seed}"]
    
    if args.task == 'sequence_tagging_en':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer SGD --lr 0.1"], 
                       ["--batch_size 8", 
                        "--batch_size 16", 
                        "--batch_size 32"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       # ["--grad_clip -1", "--grad_clip 5"], 
                       # ["", "--use_locked_drop"], 
                       ["fs"], 
                       ["", "--use_elmo"], 
                       ["", "--use_flair"], 
                       ["--char_arch LSTM", "--char_arch Conv"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 5e-4 --finetune_lr 1e-5", 
                        "--optimizer AdamW --lr 2e-3 --finetune_lr 1e-5"], 
                       ["--batch_size 48"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["--dec_arch SoftMax", "--dec_arch CRF"],
                       # ["", "--use_locked_drop"], 
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["", "--use_interm"], 
                       ["--bert_arch BERT_base", 
                        "--bert_arch RoBERTa_base", 
                        "--bert_arch BERT_large", 
                        "--bert_arch RoBERTa_large"]]
            
    elif args.task == 'sequence_tagging_zh':
        if args.command in ('fs', 'from_scratch'):
            options = [["--num_epochs 100"], 
                       ["--optimizer SGD --lr 0.1", 
                        "--optimizer AdamW --lr 1e-3", 
                        "--optimizer Adamax --lr 5e-3", 
                        "--optimizer Adamax --lr 1.5e-3"], 
                       ["--batch_size 8", "--batch_size 32"], 
                       ["--num_layers 1", "--num_layers 2"], 
                       ["fs"], 
                       # ["", "--use_bigram"], 
                       # ["", "--use_softword"], 
                       ["", "--use_softlexicon"]]
        else:
            options = [["--num_epochs 50"], 
                       ["--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5"], 
                       ["--batch_size 32"], 
                       ["--scheduler LinearDecayWithWarmup"], 
                       ["--dec_arch SoftMax", "--dec_arch CRF"],
                       ["ft"], 
                       ["--bert_drop_rate 0.2"], 
                       ["--bert_arch BERT_base", 
                        "--bert_arch RoBERTa_base"]]
        
        
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
        