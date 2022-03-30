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
from eznlp.dataset import Dataset
from eznlp.model import SpanAttrClassificationDecoderConfig
from eznlp.model import ExtractorConfig
from eznlp.training import Trainer, count_params, evaluate_attribute_extraction

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, build_trainer, header_format
from entity_recognition import collect_IE_assembly_config, process_IE_data


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2004', 
                            help="dataset name")
    group_data.add_argument('--pipeline_path', type=str, default="", 
                            help="path to load predicted chunks for pipeline")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--attr_decoder', type=str, default='span_attr_classification', 
                               help="attribute decoding method", choices=['span_attr_classification'])
    
    # Span-based
    group_decoder.add_argument('--agg_mode', type=str, default='max_pooling', 
                               help="aggregating mode")
    group_decoder.add_argument('--max_span_size', type=int, default=10, 
                               help="maximum span size")
    group_decoder.add_argument('--size_emb_dim', type=int, default=25, 
                               help="span size embedding dim")
    group_decoder.add_argument('--label_emb_dim', type=int, default=25, 
                               help="chunk label embedding dim")
    group_decoder.add_argument('--neg_sampling_rate', type=float, default=1.0, 
                               help="Negative sampling rate")
    return parse_to_args(parser)



def build_AE_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    decoder_config = SpanAttrClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                         neg_sampling_rate=args.neg_sampling_rate, 
                                                         max_span_size=args.max_span_size, 
                                                         size_emb_dim=args.size_emb_dim, 
                                                         label_emb_dim=args.label_emb_dim, 
                                                         in_drop_rates=drop_rates)
    
    return ExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-AE/{timestamp}"
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
    
    if len(args.pipeline_path) > 0:
        if not os.path.exists(f"{args.pipeline_path}/data-for-pipeline.pth"):
            raise RuntimeError("`pipeline_path` is specified but not existing")
        logger.info(f"Loading data from {args.pipeline_path} for pipeline...")
        train_data, dev_data, test_data = torch.load(f"{args.pipeline_path}/data-for-pipeline.pth")
    else:
        logger.info(f"Loading original data {args.dataset}...")
        train_data, dev_data, test_data = load_data(args)
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                entry['chunks_pred'] = entry['chunks']
    
    args.language = dataset2language[args.dataset]
    config = build_AE_config(args)
    train_data, dev_data, test_data = process_IE_data(train_data, dev_data, test_data, args, config)
    
    if not args.train_with_dev:
        train_set = Dataset(train_data, config, training=True)
        train_set.build_vocabs_and_dims(dev_data, test_data)
        dev_set   = Dataset(dev_data,  train_set.config, training=False)
        test_set  = Dataset(test_data, train_set.config, training=False)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=train_set.collate)
        dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, collate_fn=dev_set.collate)
    else:
        train_set = Dataset(train_data + dev_data, config, training=True)
        train_set.build_vocabs_and_dims(test_data)
        dev_set   = Dataset([],        train_set.config, training=False)
        test_set  = Dataset(test_data, train_set.config, training=False)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=train_set.collate)
        dev_loader   = None
    
    logger.info(train_set.summary)
    
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
    evaluate_attribute_extraction(trainer, dev_set, batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_attribute_extraction(trainer, test_set, batch_size=args.batch_size)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
