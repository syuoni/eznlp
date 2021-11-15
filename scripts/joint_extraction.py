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
from eznlp.model import EncoderConfig
from eznlp.model import SequenceTaggingDecoderConfig, SpanClassificationDecoderConfig, BoundarySelectionDecoderConfig
from eznlp.model import SpanAttrClassificationDecoderConfig
from eznlp.model import SpanRelClassificationDecoderConfig
from eznlp.model import JointExtractionDecoderConfig
from eznlp.model import ExtractorConfig
from eznlp.training import Trainer, count_params, evaluate_joint_extraction

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, build_trainer, header_format
from entity_recognition import collect_IE_assembly_config, process_IE_data


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2004', 
                            help="dataset name")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging', 
                               help="chunk decoding method", choices=['sequence_tagging', 'span_classification', 'boundary_selection'])
    group_decoder.add_argument('--attr_decoder', type=str, default='None', 
                               help="attribute decoding method", choices=['None', 'span_attr_classification'])
    group_decoder.add_argument('--rel_decoder', type=str, default='span_rel_classification', 
                               help="relation decoding method", choices=['None', 'span_rel_classification'])
    
    # Loss
    group_decoder.add_argument('--fl_gamma', type=float, default=0.0, 
                               help="Focal Loss gamma")
    group_decoder.add_argument('--sl_epsilon', type=float, default=0.0, 
                               help="Label smoothing loss epsilon")
    
    # Sequence tagging
    group_decoder.add_argument('--scheme', type=str, default='BIOES', 
                               help="sequence tagging scheme", choices=['BIOES', 'BIO2'])
    group_decoder.add_argument('--no_crf', dest='use_crf', default=True, action='store_false', 
                               help="whether to use CRF")
    
    # Span-based
    group_decoder.add_argument('--agg_mode', type=str, default='max_pooling', 
                               help="aggregating mode")
    group_decoder.add_argument('--num_neg_chunks', type=int, default=100, 
                               help="number of sampling negative chunks")
    group_decoder.add_argument('--max_span_size', type=int, default=10, 
                               help="maximum span size")
    group_decoder.add_argument('--ck_size_emb_dim', type=int, default=25, 
                               help="span size embedding dim")
    group_decoder.add_argument('--ck_label_emb_dim', type=int, default=25, 
                               help="chunk label embedding dim")
    group_decoder.add_argument('--num_neg_relations', type=int, default=100, 
                               help="number of sampling negative relations")
    group_decoder.add_argument('--max_pair_distance', type=int, default=100, 
                               help="maximum pair distance")
    
    # Boundary selection
    group_decoder.add_argument('--no_biaffine', dest='use_biaffine', default=True, action='store_false', 
                               help="whether to use biaffine")
    group_decoder.add_argument('--affine_arch', type=str, default='FFN', 
                               help="affine encoder architecture")
    group_decoder.add_argument('--sb_epsilon', type=float, default=0.0, 
                               help="Boundary smoothing loss epsilon")
    return parse_to_args(parser)



def build_joint_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    if args.ck_decoder == 'sequence_tagging':
        ck_decoder_config = SequenceTaggingDecoderConfig(scheme=args.scheme, 
                                                         use_crf=args.use_crf, 
                                                         fl_gamma=args.fl_gamma,
                                                         sl_epsilon=args.sl_epsilon, 
                                                         in_drop_rates=drop_rates)
    elif args.ck_decoder == 'span_classification':
        ck_decoder_config = SpanClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                            fl_gamma=args.fl_gamma,
                                                            sl_epsilon=args.sl_epsilon, 
                                                            num_neg_chunks=args.num_neg_chunks, 
                                                            max_span_size=args.max_span_size, 
                                                            size_emb_dim=args.ck_size_emb_dim, 
                                                            in_drop_rates=drop_rates)
    elif args.ck_decoder == 'boundary_selection':
        ck_decoder_config = BoundarySelectionDecoderConfig(use_biaffine=args.use_biaffine, 
                                                           affine=EncoderConfig(arch=args.affine_arch, hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2), 
                                                           fl_gamma=args.fl_gamma,
                                                           sl_epsilon=args.sl_epsilon, 
                                                           sb_epsilon=args.sb_epsilon,
                                                           hid_drop_rates=drop_rates)
    
    if args.attr_decoder == 'None':
        attr_decoder_config = None
    elif args.attr_decoder == 'span_attr_classification':
        attr_decoder_config = SpanAttrClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                                  max_span_size=args.max_span_size, 
                                                                  ck_size_emb_dim=args.ck_size_emb_dim, 
                                                                  ck_label_emb_dim=args.ck_label_emb_dim, 
                                                                  in_drop_rates=drop_rates)
    
    if args.rel_decoder == 'None':
        rel_decoder_config = None
    elif args.rel_decoder == 'span_rel_classification':
        rel_decoder_config = SpanRelClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                                fl_gamma=args.fl_gamma,
                                                                sl_epsilon=args.sl_epsilon, 
                                                                num_neg_relations=args.num_neg_relations, 
                                                                max_span_size=args.max_span_size, 
                                                                max_pair_distance=args.max_pair_distance, 
                                                                ck_size_emb_dim=args.ck_size_emb_dim, 
                                                                ck_label_emb_dim=args.ck_label_emb_dim, 
                                                                in_drop_rates=drop_rates)
    
    decoder_config = JointExtractionDecoderConfig(ck_decoder=ck_decoder_config, 
                                                  attr_decoder = attr_decoder_config,
                                                  rel_decoder=rel_decoder_config)
    return ExtractorConfig(**collect_IE_assembly_config(args) , decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-Joint/{timestamp}"
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
    
    train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]
    config = build_joint_config(args)
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
    evaluate_joint_extraction(trainer, dev_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), eval_chunk_type_for_relation=True, batch_size=args.batch_size)
    evaluate_joint_extraction(trainer, dev_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), eval_chunk_type_for_relation=False, batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_joint_extraction(trainer, test_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), eval_chunk_type_for_relation=True, batch_size=args.batch_size)
    evaluate_joint_extraction(trainer, test_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), eval_chunk_type_for_relation=False, batch_size=args.batch_size)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
