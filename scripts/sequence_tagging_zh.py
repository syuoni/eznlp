# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import random
import pdb
import logging
import pprint
import numpy as np
import torch
import transformers

from eznlp import auto_device
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig, SoftLexiconConfig
from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.pretrained import Vectors, BertLikeConfig
from eznlp.training.utils import LRLambda
from eznlp.training.utils import count_params, collect_params, check_param_groups

from utils import load_data, evaluate_sequence_tagging, header_format


def parse_arguments(parser: argparse.ArgumentParser):
    group_debug = parser.add_argument_group('debug')
    group_debug.add_argument('--pdb', default=False, action='store_true', 
                             help="whether to use pdb for debug")
    group_debug.add_argument('--no_log_terminal', dest='log_terminal', default=True, action='store_false', 
                             help="whether log to terminal")
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='ResumeNER', 
                            help="dataset name")
    group_data.add_argument('--scheme', type=str, default='BIOES', 
                            help="sequence tagging scheme", choices=['BIOES', 'BIO2'])
    
    group_train = parser.add_argument_group('training hyper-parameters')
    group_train.add_argument('--seed', type=int, default=515, 
                             help="random seed")
    group_train.add_argument('--use_amp', default=False, action='store_true', 
                             help="whether to use amp")
    group_train.add_argument('--num_epochs', type=int, default=100, 
                             help="number of epochs")
    group_train.add_argument('--batch_size', type=int, default=10, 
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
                             help='scheduler', choices=['None', 'ReduceLROnPlateau', 'LinearDecayWithWarmup'])
    
    group_model = parser.add_argument_group('model configurations')
    group_model.add_argument('--hid_dim', type=int, default=200, 
                             help="hidden dim")
    group_model.add_argument('--num_layers', type=int, default=1, 
                             help="number of encoder layers")
    group_model.add_argument('--drop_rate', type=float, default=0.5, 
                             help="dropout rate")
    group_model.add_argument('--use_locked_drop', default=False, action='store_true', 
                             help="whether to use locked dropout")
    group_model.add_argument('--dec_arch', type=str, default='CRF', 
                             help="decoder architecture")
    
    subparsers = parser.add_subparsers(dest='command', help="sub-commands")
    parser_fs = subparsers.add_parser('from_scratch', aliases=['fs'], 
                                      help="train from scratch, or with freezed pretrained models")
    parser_fs.add_argument('--emb_dim', type=int, default=50, 
                           help="embedding dim")
    parser_fs.add_argument('--emb_freeze', default=False, action='store_true', 
                           help="whether to freeze embedding weights")
    parser_fs.add_argument('--use_interm1', default=False, action='store_true', 
                           help="whether to use intermediate1")
    parser_fs.add_argument('--use_bigram', default=False, action='store_true', 
                           help="whether to use bigram")
    parser_fs.add_argument('--use_softword', default=False, action='store_true', 
                           help="whether to use softword")
    parser_fs.add_argument('--use_softlexicon', default=False, action='store_true', 
                           help="whether to use softlexicon")
    
    parser_ft = subparsers.add_parser('finetune', aliases=['ft'], 
                                      help="train by finetuning pretrained models")
    parser_ft.add_argument('--bert_arch', type=str, default='BERT_base', 
                           help="bert-like architecture")
    parser_ft.add_argument('--bert_drop_rate', type=float, default=0.2, 
                           help="dropout rate for BERT")
    parser_ft.add_argument('--use_interm2', default=False, action='store_true', 
                           help="whether to use intermediate2")
    
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    args.grad_clip = None if args.grad_clip < 0 else args.grad_clip
    return args
    

def build_config(args: argparse.Namespace):
    if args.use_locked_drop:
        drop_rates = (0.0, 0.05, args.drop_rate)
    else:
        drop_rates = (args.drop_rate, 0.0, 0.0)
        
    decoder_config = SequenceTaggingDecoderConfig(arch=args.dec_arch, scheme=args.scheme, in_drop_rates=drop_rates)
    
    if args.command in ('from_scratch', 'fs'):
        giga_uni = Vectors.load("assets/vectors/gigaword_chn.all.a2b.uni.ite50.vec", encoding='utf-8')
        ohots_config = ConfigDict({'text': OneHotConfig(field='text', vectors=giga_uni, emb_dim=50, freeze=args.emb_freeze)})
        if args.use_bigram:
            giga_bi = Vectors.load("assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec", encoding='utf-8')
            ohots_config['bigram'] = OneHotConfig(field='bigram', vectors=giga_bi, emb_dim=50, freeze=args.emb_freeze)
        
        if args.use_softword:
            mhots_config = ConfigDict({'softword': MultiHotConfig(field='softword', use_emb=False)})
        else:
            mhots_config = None
        
        if args.use_softlexicon:
            ctb50 = Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
            nested_ohots_config = ConfigDict({'softlexicon': SoftLexiconConfig(vectors=ctb50, emb_dim=50, freeze=args.emb_freeze)})
        else:
            nested_ohots_config = None
            
        if args.use_interm1:
            interm1_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        else:
            interm1_config = None
            
        interm2_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        bert_like_config = None
        
    elif args.command in ('finetune', 'ft'):
        ohots_config = None
        mhots_config = None
        nested_ohots_config = None
        interm1_config = None
        
        if args.use_interm2:
            interm2_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        else:
            interm2_config = None
        
        if args.bert_arch.startswith('BERT'):
            PATH = "assets/transformers/hfl/chinese-bert-wwm-ext"
            tokenizer = transformers.AutoTokenizer.from_pretrained(PATH)
            bert = transformers.AutoModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, 
                                                          attention_probs_dropout_prob=args.bert_drop_rate)
            bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert, arch=args.bert_arch, 
                                              freeze=False, use_truecase=True)
        elif args.bert_arch.startswith('RoBERTa'):
            PATH = "assets/transformers/hfl/chinese-roberta-wwm-ext"
            tokenizer = transformers.AutoTokenizer.from_pretrained(PATH, add_prefix_space=True)
            roberta = transformers.AutoModel.from_pretrained(PATH, hidden_dropout_prob=args.bert_drop_rate, 
                                                             attention_probs_dropout_prob=args.bert_drop_rate)
            bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=roberta, arch=args.bert_arch, 
                                              freeze=False, use_truecase=False)
            
    else:
        raise Exception("No sub-command specified")
        
    return SequenceTaggerConfig(ohots=ohots_config, 
                                mhots=mhots_config, 
                                nested_ohots=nested_ohots_config, 
                                intermediate1=interm1_config, 
                                bert_like=bert_like_config, 
                                intermediate2=interm2_config, 
                                decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}/{timestamp}"
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
        
    handlers=[logging.FileHandler(f"{save_path}/training.log")]
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
        
    train_data, dev_data, test_data = load_data(args)
    config = build_config(args)
    
    train_set = SequenceTaggingDataset(train_data, config)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    dev_set   = SequenceTaggingDataset(dev_data,  train_set.config)
    test_set  = SequenceTaggingDataset(test_data, train_set.config)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate)
    
    
    logger.info(header_format("Building", sep='-'))
    tagger = config.instantiate().to(device)
    count_params(tagger)
    
    logger.info(header_format("Training", sep='-'))
    param_groups = [{'params': tagger.pretrained_parameters(), 'lr': args.finetune_lr}]
    param_groups.append({'params': collect_params(tagger, param_groups), 'lr': args.lr})
    assert check_param_groups(tagger, param_groups)
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
        lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=len(train_loader)*num_warmup_epochs, 
                                                         num_total_steps=len(train_loader)*args.num_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        
    trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, scheduler=scheduler, schedule_by_step=schedule_by_step,
                                     device=device, grad_clip=args.grad_clip, use_amp=args.use_amp)
    if args.pdb: 
        pdb.set_trace()
        
    def save_callback(model):
        torch.save(model, f"{save_path}/{args.scheme}-{config.name}.pth")
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    logger.info(header_format("Evaluating", sep='-'))
    tagger = torch.load(f"{save_path}/{args.scheme}-{config.name}.pth", map_location=device)
    trainer = SequenceTaggingTrainer(tagger, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_sequence_tagging(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_sequence_tagging(trainer, test_set)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
    
    