# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import pdb
import logging
import pprint
import torch

from eznlp import auto_device
from eznlp.dataset import Dataset
from eznlp.model import SequenceTaggingDecoderConfig, SpanClassificationDecoderConfig, RelationClassificationDecoderConfig, JointDecoderConfig
from eznlp.model import ModelConfig
from eznlp.training import Trainer
from eznlp.training.utils import count_params
from eznlp.training.evaluation import evaluate_joint_er_re

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, build_trainer, header_format
from entity_recognition import collect_IE_assembly_config


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2004', 
                            help="dataset name")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging', 
                               help="chunk decoding method", choices=['sequence_tagging', 'span_classification'])
    
    group_sequence_tagging = parser.add_argument_group('sequence tagging')
    group_sequence_tagging.add_argument('--ck_dec_arch', type=str, default='CRF', 
                                        help="decoder architecture")
    group_sequence_tagging.add_argument('--scheme', type=str, default='BIOES', 
                                        help="sequence tagging scheme", choices=['BIOES', 'BIO2'])
    
    group_span_classification = parser.add_argument_group('span classification')
    group_span_classification.add_argument('--agg_mode', type=str, default='max_pooling', 
                                           help="aggregating mode")
    group_span_classification.add_argument('--num_neg_chunks', type=int, default=100, 
                                           help="number of sampling negative chunks")
    group_span_classification.add_argument('--max_span_size', type=int, default=10, 
                                           help="maximum span size")
    group_span_classification.add_argument('--ck_size_emb_dim', type=int, default=25, 
                                           help="span size embedding dim")
    
    group_rel_classification = parser.add_argument_group('relation classification')
    group_rel_classification.add_argument('--num_neg_relations', type=int, default=100, 
                                          help="number of sampling negative relations")
    group_rel_classification.add_argument('--ck_label_emb_dim', type=int, default=25, 
                                          help="chunk label embedding dim")
    
    return parse_to_args(parser)



def build_JERRE_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    if args.ck_dec_arch.lower() in ('crf', 'softmax'):
        ck_decoder_config = SequenceTaggingDecoderConfig(arch=args.ck_dec_arch, 
                                                         scheme=args.scheme, 
                                                         in_drop_rates=drop_rates)
    else:
        ck_decoder_config = SpanClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                            num_neg_chunks=args.num_neg_chunks, 
                                                            max_span_size=args.max_span_size, 
                                                            size_emb_dim=args.size_emb_dim, 
                                                            in_drop_rates=drop_rates)
        
    rel_decoder_config = RelationClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                             num_neg_relations=args.num_neg_relations, 
                                                             max_span_size=args.max_span_size, 
                                                             ck_size_emb_dim=args.ck_size_emb_dim, 
                                                             ck_label_emb_dim=args.ck_label_emb_dim, 
                                                             in_drop_rates=drop_rates)
    decoder_config = JointDecoderConfig(ck_decoder=ck_decoder_config, rel_decoder=rel_decoder_config)
    return ModelConfig(**collect_IE_assembly_config(args) , decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-JERRE/{timestamp}"
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
    args.language = dataset2language[args.dataset]
    config = build_JERRE_config(args)
    
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    dev_set   = Dataset(dev_data,  train_set.config, training=False)
    test_set  = Dataset(test_data, train_set.config, training=False)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate)
    
    
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
    evaluate_joint_er_re(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_joint_er_re(trainer, test_set)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
    
    