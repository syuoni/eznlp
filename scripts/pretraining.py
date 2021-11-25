# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import pdb
import logging
import pprint
import numpy
import torch
import transformers

from eznlp import auto_device
from eznlp.io import RawTextIO
from eznlp.dataset import PreTrainingDataset
from eznlp.plm import MaskedLMConfig
from eznlp.training import MaskedLMTrainer, LRLambda, count_params

from utils import add_base_arguments, parse_to_args
from utils import header_format


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='Wikipedia_zh', 
                            help="dataset name")
    group_data.add_argument('--file_paths', type=str, nargs='*', default='data/Wikipedia/text-zh/AA/wiki_00', 
                            help="raw text file paths")
    group_data.add_argument('--prepare_data', default=False, action='store_true', 
                            help="whether to prepare data")
    group_data.add_argument('--disp_every_steps', type=int, default=1000, 
                            help="step number for display")
    
    return parse_to_args(parser)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-PT/{timestamp}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    handlers = [logging.FileHandler(f"{save_path}/training.log")]
    if args.log_terminal:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S", 
                        handlers=handlers)
    
    logger = logging.getLogger(__name__)
    logger.info(header_format("Starting", sep='='))
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    
    
    logger.info(header_format("Preparing", sep='-'))
    device = auto_device()
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        temp = torch.randn(100).to(device)
    
    PATH = "assets/transformers/bert-base-chinese"
    bert4pt = transformers.BertForMaskedLM.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate)
    tokenizer = transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True)
    
    train_data = []
    if args.prepare_data:
        io = RawTextIO(tokenizer.tokenize, max_len=510, document_sep_starts=["-DOCSTART-", "<doc", "</doc"], encoding='utf-8')
        for file_path in args.file_paths:
            data = io.read(file_path)
            io.write(data, f"{file_path}.cache")
            train_data += data
    else:
        io = RawTextIO(encoding='utf-8')
        for file_path in args.file_paths:
            data = io.read(f"{file_path}.cache")
            train_data += data
    
    config = MaskedLMConfig(bert_like=bert4pt, tokenizer=tokenizer)
    train_set = PreTrainingDataset(train_data, config)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate)
    
    logger.info(header_format("Building", sep='-'))
    model = config.instantiate().to(device)
    count_params(model)
    
    logger.info(header_format("Training", sep='-'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)  # BERT
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.980), eps=1e-6, weight_decay=0.01)  # RoBERTa
    
    num_total_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = min(10_000*args.num_grad_acc_steps, num_total_steps // 5)
    logger.info(f"Warmup steps: {num_warmup_steps:,}")
    logger.info(f"Total steps: {num_total_steps:,}")
    
    lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=num_warmup_steps, num_total_steps=num_total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    trainer = MaskedLMTrainer(model, optimizer=optimizer, scheduler=scheduler, schedule_by_step=True, num_grad_acc_steps=args.num_grad_acc_steps,
                              device=device, grad_clip=args.grad_clip, use_amp=args.use_amp)
    
    if args.pdb: 
        pdb.set_trace()
    
    trainer.train_steps(train_loader=train_loader, num_epochs=args.num_epochs, 
                        disp_every_steps=args.disp_every_steps, eval_every_steps=args.disp_every_steps*100)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
