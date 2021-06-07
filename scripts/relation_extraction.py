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
from eznlp.model import PairClassificationDecoderConfig
from eznlp.model import ModelConfig
from eznlp.training import Trainer
from eznlp.training.utils import count_params
from eznlp.training.evaluation import evaluate_relation_extraction

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, build_trainer, header_format
from entity_recognition import collect_IE_assembly_config


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2004', 
                            help="dataset name")
    group_data.add_argument('--pipeline_path', type=str, default="", 
                            help="path to load predicted chunks for pipeline")
    
    group_rel_classification = parser.add_argument_group('relation classification')
    group_rel_classification.add_argument('--agg_mode', type=str, default='max_pooling', 
                                          help="aggregating mode")
    group_rel_classification.add_argument('--criterion', type=str, default='CE', 
                                          help="decoder loss criterion")
    group_rel_classification.add_argument('--focal_gamma', type=float, default=2.0, 
                                          help="Focal Loss gamma")
    group_rel_classification.add_argument('--num_neg_relations', type=int, default=100, 
                                          help="number of sampling negative relations")
    group_rel_classification.add_argument('--max_span_size', type=int, default=10, 
                                          help="maximum span size")
    group_rel_classification.add_argument('--max_pair_distance', type=int, default=100, 
                                          help="maximum pair distance")
    group_rel_classification.add_argument('--ck_size_emb_dim', type=int, default=25, 
                                          help="chunk span size embedding dim")
    group_rel_classification.add_argument('--ck_label_emb_dim', type=int, default=25, 
                                          help="chunk label embedding dim")
    
    return parse_to_args(parser)



def build_RE_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    decoder_config = PairClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                     criterion=args.criterion,
                                                     gamma=args.focal_gamma, 
                                                     num_neg_relations=args.num_neg_relations, 
                                                     max_span_size=args.max_span_size, 
                                                     max_pair_distance=args.max_pair_distance, 
                                                     ck_size_emb_dim=args.ck_size_emb_dim, 
                                                     ck_label_emb_dim=args.ck_label_emb_dim, 
                                                     in_drop_rates=drop_rates)
    
    return ModelConfig(**collect_IE_assembly_config(args), decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-RE/{timestamp}"
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
        
    if os.path.exists(f"{args.pipeline_path}/data-with-chunks-pred.pth"):
        logger.info(f"Loading data from {args.pipeline_path} for pipeline...")
        train_data, dev_data, test_data = torch.load(f"{args.pipeline_path}/data-with-chunks-pred.pth")
    else:
        logger.info(f"Loading original data {args.dataset}...")
        train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]
    config = build_RE_config(args)
    
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
        
    torch.save(config, f"{save_path}/{config.name}-config.pth")
    def save_callback(model):
        torch.save(model, f"{save_path}/{config.name}.pth")
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    logger.info(header_format("Evaluating", sep='-'))
    model = torch.load(f"{save_path}/{config.name}.pth", map_location=device)
    trainer = Trainer(model, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_relation_extraction(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_relation_extraction(trainer, test_set)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
    