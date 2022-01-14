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

from eznlp import auto_device
from eznlp.token import TokenSequence
from eznlp.dataset import Dataset
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, EncoderConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model import TextClassificationDecoderConfig
from eznlp.model import ClassifierConfig
from eznlp.model.bert_like import truncate_for_bert_like
from eznlp.training import Trainer, count_params, evaluate_text_classification

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, load_vectors, build_trainer, header_format


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='imdb', 
                            help="dataset name")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--agg_mode', type=str, default='multiplicative_attention', 
                               help="aggregating mode")
    
    return parse_to_args(parser)



def collect_TC_assembly_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    if args.emb_dim > 0:
        vectors = load_vectors(args.language, args.emb_dim)
        ohots_config = ConfigDict({'text': OneHotConfig(field='text', min_freq=5, vectors=vectors, emb_dim=args.emb_dim, freeze=args.emb_freeze)})
    else:
        ohots_config = None
    
    if args.use_interm1:
        interm1_config = EncoderConfig(arch=args.enc_arch, hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
    else:
        interm1_config = None
    
    if args.use_interm2:
        interm2_config = EncoderConfig(arch=args.enc_arch, hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates)
    else:
        interm2_config = None
    
    if args.language.lower() == 'english' and args.use_elmo:
        elmo_config = ELMoConfig(elmo=load_pretrained('elmo'))
    else:
        elmo_config = None
    
    if args.language.lower() == 'english' and args.use_flair:
        flair_fw_lm, flair_bw_lm = load_pretrained('flair')
        flair_fw_config, flair_bw_config = FlairConfig(flair_lm=flair_fw_lm), FlairConfig(flair_lm=flair_bw_lm)
        if interm2_config is not None:
            interm2_config.in_proj = True
    else:
        flair_fw_config, flair_bw_config = None, None
    
    if args.bert_arch.lower() != 'none':
        # Uncased tokenizer for text classification
        bert_like, tokenizer = load_pretrained(args.bert_arch, args, cased=False)
        bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_like, arch=args.bert_arch, freeze=False, 
                                          paired_inputs=args.paired_inputs, 
                                          use_truecase='cased' in os.path.basename(bert_like.name_or_path).split('-'))
    else:
        bert_like_config = None
    
    return {'ohots': ohots_config, 
            'intermediate1': interm1_config, 
            'elmo': elmo_config, 
            'flair_fw': flair_fw_config, 
            'flair_bw': flair_bw_config, 
            'bert_like': bert_like_config, 
            'intermediate2': interm2_config, }


def build_TC_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    decoder_config = TextClassificationDecoderConfig(agg_mode=args.agg_mode, in_drop_rates=drop_rates)
    return ClassifierConfig(**collect_TC_assembly_config(args), decoder=decoder_config)


def process_TC_data(train_data, dev_data, test_data, args, config):
    # Truncate too long sentences
    if config.bert_like is not None:
        train_data = truncate_for_bert_like(train_data, config.bert_like.tokenizer, verbose=args.log_terminal)
        dev_data   = truncate_for_bert_like(dev_data,   config.bert_like.tokenizer, verbose=args.log_terminal)
        test_data  = truncate_for_bert_like(test_data,  config.bert_like.tokenizer, verbose=args.log_terminal)
        
    elif args.dataset in ('ChnSentiCorp', 'THUCNews_10'):
        # Too long sentences even for RNN
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                if len(entry['tokens']) > 1200:
                    entry['tokens'] = entry['tokens'][:300] + entry['tokens'][-900:]
    
    if config.bert_like is not None and config.bert_like.paired_inputs:
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                entry['tokens'] = entry['tokens'] + TokenSequence.from_tokenized_text([config.bert_like.tokenizer.sep_token]) + entry['paired_tokens']
    
    return train_data, dev_data, test_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-TC/{timestamp}"
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
        
    train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]
    args.paired_inputs = ('paired_tokens' in train_data[0])
    # train_data, dev_data, test_data = train_data[:1000], dev_data[:1000], test_data[:1000]
    config = build_TC_config(args)
    train_data, dev_data, test_data = process_TC_data(train_data, dev_data, test_data, args, config)
    
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data)
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
    
    torch.save(config, f"{save_path}/{config.name}-config.pth")
    def save_callback(model):
        torch.save(model, f"{save_path}/{config.name}.pth")
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    logger.info(header_format("Evaluating", sep='-'))
    model = torch.load(f"{save_path}/{config.name}.pth", map_location=device)
    trainer = Trainer(model, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_text_classification(trainer, dev_set, batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_text_classification(trainer, test_set, batch_size=args.batch_size)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
