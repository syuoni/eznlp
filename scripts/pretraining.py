# -*- coding: utf-8 -*-
import os
import sys
import argparse
import collections
import datetime
import pdb
import glob
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


"""See: https://github.com/tczhangzhi/pytorch-distributed

NCCL_P2P_DISABLE=1 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 scripts/pretraining.py @scripts/options/pt_bert.opt
"""


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='HwaMei', 
                            help="dataset name")
    group_data.add_argument('--folder_path', type=str, default='../HwaMei-text', 
                            help="the folder path of prepared pretraining data")
    group_data.add_argument('--no_hwamei', dest='use_hwamei', default=True, action='store_false', 
                            help="whether to use HwaMei data")
    group_data.add_argument('--use_wiki', default=False, action='store_true', 
                            help="whether to use wiki data")
    group_data.add_argument('--use_medical_corpora', default=False, action='store_true', 
                            help="whether to use medical corpora")
    group_data.add_argument('--use_medical_dialogue', default=False, action='store_true', 
                            help="whether to use medical dialogue corpora")
    group_data.add_argument('--use_common_corpora', default=False, action='store_true', 
                            help="whether to use common corpora")
    
    group_pretrain = parser.add_argument_group('pretrain')
    group_pretrain.add_argument('--vocab_fix', default=False, action='store_true', 
                                help="whether to load the vocab fixed BERT")
    group_pretrain.add_argument('--masking_rate', type=float, default=0.15, 
                                help="masking rate")
    group_pretrain.add_argument('--masking_rate_dev', type=float, default=0.0, 
                                help="masking rate deviation")
    group_pretrain.add_argument('--use_wwm', default=False, action='store_true', 
                                help="whether to use whole word masking")
    group_pretrain.add_argument('--use_ngram', default=False, action='store_true', 
                                help="whether to use N-gram masking")
    group_pretrain.add_argument('--paired_task', type=str, default='None', 
                                help="paired task", choices=['None', 'NSP', 'SOP'])
    
    group_train = parser.add_argument_group('training etc')
    group_train.add_argument('--disp_every_steps', type=int, default=1000, 
                             help="step number for display")
    group_train.add_argument('--local_rank', type=int, default=-1, 
                             help='node rank for distributed training')
    group_train.add_argument('--ddp_backend', type=str, default='nccl', 
                             help='DDP backend', choices=['nccl', 'gloo'])
    
    return parse_to_args(parser)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    is_main_rank = (args.local_rank <= 0)
    use_ddp = (args.local_rank >= 0)
    args.log_terminal = (args.log_terminal and is_main_rank)
    
    if is_main_rank:
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
    logger.disabled = (not is_main_rank)
    logger.info(header_format("Starting", sep='='))
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    
    
    logger.info(header_format("Preparing", sep='-'))
    
    if use_ddp:
        torch.distributed.init_process_group(backend=args.ddp_backend)
        device = torch.device('cuda', args.local_rank)
    else:
        device = auto_device()
    
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        temp = torch.randn(100).to(device)
    
    if args.vocab_fix:
        PATH = "assets/transformers/syuoni/bert-base-chinese-vf"
    else:
        PATH = "assets/transformers/bert-base-chinese"
    if args.paired_task.lower() == 'none':
        bert4pt = transformers.BertForMaskedLM.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate)
    else:
        bert4pt = transformers.BertForPreTraining.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, attention_probs_dropout_prob=args.bert_drop_rate)
    tokenizer = transformers.BertTokenizer.from_pretrained(PATH, model_max_length=512, do_lower_case=True)
    
    file_paths = []
    if args.use_hwamei:
        file_paths += glob.glob(f"{args.folder_path}/hm/*.cache")
    if args.use_wiki:
        file_paths += glob.glob(f"{args.folder_path}/wiki-zh/*/*.cache")
    if args.use_medical_corpora:
        file_paths += glob.glob(f"{args.folder_path}/medical-books/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/baikemy/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/med-prescription/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/tcmed-literature/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/wudao-med/*.cache")
    if args.use_medical_dialogue:
        file_paths += glob.glob(f"{args.folder_path}/med-dialogue/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/med-qa/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/cmc-dialogue/*.cache")
    if args.use_common_corpora:
        file_paths += glob.glob(f"{args.folder_path}/chatbot/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/weixin/*.cache")
        file_paths += glob.glob(f"{args.folder_path}/thucnews/pretraining/*.cache")
    
    assert len(file_paths) > 0
    sources = collections.Counter([fn.replace(f"{args.folder_path}/", "").split("/")[0] for fn in file_paths])
    for src, num in sources.items():
        logger.info(f"Text files of {src}: {num}")
    
    io = RawTextIO(encoding='utf-8', verbose=args.log_terminal)
    train_data = []
    for fn in file_paths:
        train_data += io.read(fn)
    
    config = MaskedLMConfig(bert_like=bert4pt, tokenizer=tokenizer, 
                            masking_rate=args.masking_rate, masking_rate_dev=args.masking_rate_dev, 
                            use_wwm=args.use_wwm, ngram_weights=(0.5, 0.35, 0.15) if args.use_ngram else (1.0, ), 
                            paired_task=args.paired_task)
    train_set = PreTrainingDataset(train_data, config)
    
    logger.info(train_set.summary)
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=not use_ddp, collate_fn=train_set.collate, sampler=train_sampler)
    
    logger.info(header_format("Building", sep='-'))
    model = config.instantiate().to(device)
    count_params(model)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    logger.info(header_format("Training", sep='-'))
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)  # BERT
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.980), eps=1e-6, weight_decay=0.01)  # RoBERTa
    
    num_total_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = min(10_000*args.num_grad_acc_steps, num_total_steps//5)
    logger.info(f"Warmup steps: {num_warmup_steps:,}")
    logger.info(f"Total steps: {num_total_steps:,}")
    
    lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=num_warmup_steps, num_total_steps=num_total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    trainer = MaskedLMTrainer(model, optimizer=optimizer, scheduler=scheduler, schedule_by_step=True, num_grad_acc_steps=args.num_grad_acc_steps,
                              device=device, non_blocking=use_ddp, grad_clip=args.grad_clip, use_amp=args.use_amp)
    
    if args.pdb: 
        pdb.set_trace()
    
    trainer.train_steps(train_loader=train_loader, num_epochs=args.num_epochs, 
                        disp_every_steps=args.disp_every_steps, eval_every_steps=args.disp_every_steps*100)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if is_main_rank:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
