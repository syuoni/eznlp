# -*- coding: utf-8 -*-
import itertools
import argparse
import subprocess
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='conll2003', 
                        help="dataset name")
    parser.add_argument('--command', type=str, default='fs', 
                        help="sub-commands")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    
    COMMAND = "python scripts/sequence_tagging_en.py"
    COMMAND = " ".join([COMMAND, f"--dataset {args.dataset}"])
    
    
    if args.command == 'fs':
        options = [["--num_epochs 100"], 
                   ["--optimizer SGD --lr 0.1", 
                    "--optimizer SGD --lr 0.2", 
                    "--optimizer SGD --lr 0.05",
                    "--optimizer AdamW --lr 1e-3"], 
                   # ["--batch_size 10", "--batch_size 64"], 
                   ["--num_layers 1", "--num_layers 2"], 
                   # ["", "--use_elmo"], 
                   # ["", "--use_flair"], 
                   ["fs"], 
                   ["--char_arch LSTM", "--char_arch CNN"]]
    else:
        options = [["--num_epochs 50"], 
                   ["--batch_size 32"], 
                   ["--optimizer AdamW --lr 1e-4 --finetune_lr 2e-5", 
                    "--optimizer AdamW --lr 2e-5 --finetune_lr 2e-5", 
                    "--optimizer AdamW --lr 1e-5 --finetune_lr 1e-5", 
                    "--optimizer SGD --lr 1e-3 --finetune_lr 1e-3"], 
                   ["--scheduler LinearDecayWithWarmup"], 
                   ["ft"], 
                   ["", "--use_roberta"]]
        
    commands = []
    for curr_option in itertools.product(*options):
        curr_option = [x for x in curr_option if len(x) > 0]
        curr_command = " ".join([COMMAND, *curr_option])
        commands.append(curr_command)
        
    logger.warning(f"There are {len(commands)} scripts to run...")
    logger.warning("\n".join(commands))
    
    
    for curr_command in commands:
        logger.warning(f"Starting: {curr_command}")
        subprocess.call(curr_command.split())
        logger.warning(f"Ending: {curr_command}")
        
        