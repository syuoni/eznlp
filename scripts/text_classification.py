# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import random
import pdb
import logging
import pprint
import numpy
import torch

from eznlp import auto_device
from eznlp.vectors import Vectors, GloVe
from eznlp.dataset import Dataset
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, EncoderConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model import TextClassificationDecoderConfig
from eznlp.model import ModelConfig
from eznlp.model.bert_like import truncate_for_bert_like
from eznlp.training import Trainer
from eznlp.training.utils import count_params
from eznlp.training.evaluation import evaluate_text_classification

from utils import load_data, dataset2language, load_pretrained, build_trainer, header_format


def parse_arguments(parser: argparse.ArgumentParser):
    group_debug = parser.add_argument_group('debug')
    group_debug.add_argument('--pdb', default=False, action='store_true', 
                             help="whether to use pdb for debug")
    group_debug.add_argument('--no_log_terminal', dest='log_terminal', default=True, action='store_false', 
                             help="whether log to terminal")
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='imdb', 
                            help="dataset name")
    
    group_train = parser.add_argument_group('training hyper-parameters')
    group_train.add_argument('--seed', type=int, default=515, 
                             help="random seed")
    group_train.add_argument('--use_amp', default=False, action='store_true', 
                             help="whether to use amp")
    group_train.add_argument('--num_epochs', type=int, default=50, 
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
    group_model.add_argument('--agg_mode', type=str, default='multiplicative_attention', 
                             help="aggregating mode")
    
    subparsers = parser.add_subparsers(dest='command', help="sub-commands")
    parser_fs = subparsers.add_parser('from_scratch', aliases=['fs'], 
                                      help="train from scratch, or with freezed pretrained models")
    parser_fs.add_argument('--emb_dim', type=int, default=100, 
                           help="embedding dim")
    parser_fs.add_argument('--emb_freeze', default=False, action='store_true', 
                           help="whether to freeze embedding weights")
    parser_fs.add_argument('--use_interm1', default=False, action='store_true', 
                           help="whether to use intermediate1")
    parser_fs.add_argument('--use_elmo', default=False, action='store_true', 
                           help="whether to use ELMo")
    parser_fs.add_argument('--use_flair', default=False, action='store_true', 
                           help="whether to use Flair")
    
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
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    args.grad_clip = None if args.grad_clip < 0 else args.grad_clip
    return args
    

def build_config(args: argparse.Namespace):
    if args.use_locked_drop:
        drop_rates = (0.0, 0.05, args.drop_rate)
    else:
        drop_rates = (args.drop_rate, 0.0, 0.0)
        
        
    if args.command in ('from_scratch', 'fs'):
        if args.language.lower() == 'english' and args.emb_dim in (50, 100, 200):
            vectors = GloVe(f"assets/vectors/glove.6B.{args.emb_dim}d.txt")
        elif args.language.lower() == 'english' and args.emb_dim == 300:
            vectors = GloVe("assets/vectors/glove.840B.300d.txt")
        elif args.language.lower() == 'chinese' and args.emb_dim == 50:
            vectors = Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
        elif args.language.lower() == 'chinese' and args.emb_dim == 200:
            vectors = Vectors.load("assets/vectors/tencent/Tencent_AILab_ChineseEmbedding.txt", encoding='utf-8', skiprows=0)
        else:
            vectors = None
        ohots_config = ConfigDict({'text': OneHotConfig(field='text', min_freq=5, vectors=vectors, emb_dim=args.emb_dim, freeze=args.emb_freeze)})
        
        if args.use_interm1:
            interm1_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        else:
            interm1_config = None
            
        interm2_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        
        if args.language.lower() == 'english' and args.use_elmo:
            elmo_config = ELMoConfig(elmo=load_pretrained('elmo'))
        else:
            elmo_config = None
            
        if args.language.lower() == 'english' and args.use_flair:
            flair_fw_lm, flair_bw_lm = load_pretrained('flair')
            flair_fw_config, flair_bw_config = FlairConfig(flair_lm=flair_fw_lm), FlairConfig(flair_lm=flair_bw_lm)
            interm2_config.in_proj = True
        else:
            flair_fw_config, flair_bw_config = None, None
            
        bert_like_config = None
        
    elif args.command in ('finetune', 'ft'):
        ohots_config = None
        interm1_config = None
        
        if args.use_interm2:
            interm2_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
        else:
            interm2_config = None
        
        elmo_config = None
        flair_fw_config, flair_bw_config = None, None
        
        # Uncased tokenizer for text classification
        bert_like, tokenizer = load_pretrained(args.bert_arch, args, cased=False)
        bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_like, arch=args.bert_arch, 
                                          freeze=False, use_truecase=args.bert_arch.startswith('BERT'))
    else:
        raise Exception("No sub-command specified")
        
    decoder_config = TextClassificationDecoderConfig(agg_mode=args.agg_mode, in_drop_rates=drop_rates)
    return ModelConfig(ohots=ohots_config, 
                       intermediate1=interm1_config, 
                       elmo=elmo_config, 
                       flair_fw=flair_fw_config, 
                       flair_bw=flair_bw_config, 
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
        temp = torch.randn(100).to(device)
        
    train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]
    # train_data, dev_data, test_data = train_data[:1000], dev_data[:1000], test_data[:1000]
    config = build_config(args)
    
    if args.command in ('finetune', 'ft'):
        train_data = truncate_for_bert_like(train_data, config.bert_like.tokenizer, verbose=args.log_terminal)
        dev_data   = truncate_for_bert_like(dev_data,   config.bert_like.tokenizer, verbose=args.log_terminal)
        test_data  = truncate_for_bert_like(test_data,  config.bert_like.tokenizer, verbose=args.log_terminal)
        
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    dev_set   = Dataset(dev_data,  train_set.config, training=False)
    test_set  = Dataset(test_data, train_set.config, training=False)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, collate_fn=dev_set.collate)
    
    
    logger.info(header_format("Building", sep='-'))
    model = config.instantiate().to(device)
    count_params(model)
    
    logger.info(header_format("Training", sep='-'))
    trainer = build_trainer(model, device, len(train_loader), args)
    if args.pdb: 
        pdb.set_trace()
        
    torch.save(config, f"{save_path}/{config.name}.config")
    def save_callback(model):
        torch.save(model, f"{save_path}/{config.name}.pth")
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    logger.info(header_format("Evaluating", sep='-'))
    model = torch.load(f"{save_path}/{config.name}.pth", map_location=device)
    trainer = Trainer(model, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_text_classification(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_text_classification(trainer, test_set)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
    
    