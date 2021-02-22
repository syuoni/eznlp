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
    parser.add_argument('--dataset', type=str, default='conll2003', 
                        help="dataset name")
    parser.add_argument('--command', type=str, default='fs', 
                        help="sub-commands")
    parser.add_argument('--num_workers', type=int ,default=0, 
                        help="number of processes to run")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = "python scripts/sequence_tagging_en.py"
    COMMAND = " ".join([COMMAND, f"--dataset {args.dataset}"])
    
    
    if args.command in ('fs', 'from_scratch'):
        options = [["--num_epochs 100"], 
                   ["--optimizer SGD --lr 0.1", 
                    "--optimizer SGD --lr 0.05"], 
                   ["--batch_size 10", 
                    "--batch_size 32", 
                    "--batch_size 64"], 
                   ["--num_layers 1", "--num_layers 2"], 
                   ["--grad_clip -1", "--grad_clip 5"], 
                   # ["", "--use_elmo"], 
                   # ["", "--use_flair"], 
                   ["fs"], 
                   ["--char_arch LSTM", "--char_arch CNN"]]
    else:
        options = [["--num_epochs 50"], 
                   ["--optimizer AdamW --lr 1e-3 --finetune_lr 1e-5", 
                    "--optimizer AdamW --lr 1e-4 --finetune_lr 1e-5", 
                    "--optimizer AdamW --lr 1e-5 --finetune_lr 1e-5", 
                    "--optimizer AdamW --lr 1e-4 --finetune_lr 2e-5", 
                    "--optimizer SGD --lr 0.1 --finetune_lr 0.01"], 
                   ["--batch_size 10", 
                    "--batch_size 32", 
                    "--batch_size 64"], 
                   ["--scheduler LinearDecayWithWarmup"], 
                   ["--dec_arch SoftMax", "--dec_arch CRF"],
                   ["ft"], 
                   ["", "--use_roberta"]]
    
    if args.num_workers > 0:
        options.insert(0, ["--no_log_terminal"])
        
    commands = []
    for curr_option in itertools.product(*options):
        curr_option = [x for x in curr_option if len(x) > 0]
        curr_command = " ".join([COMMAND, *curr_option])
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
            # Ensure auto-device allocated before the next process start...
            time.sleep(20)
        pool.close()
        pool.join()
        