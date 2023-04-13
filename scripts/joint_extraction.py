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
    group_data.add_argument('--save_preds', default=False, action='store_true', 
                            help="whether to save predictions on the test split (typically in case without ground truth)")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging', 
                               help="chunk decoding method", choices=['sequence_tagging', 'span_classification', 'boundary_selection'])
    group_decoder.add_argument('--attr_decoder', type=str, default='None', 
                               help="attribute decoding method", choices=['None', 'span_attr_classification'])
    group_decoder.add_argument('--rel_decoder', type=str, default='span_rel_classification', 
                               help="relation decoding method", choices=['None', 'span_rel_classification'])
    
    # Loss
    group_decoder.add_argument('--fl_gamma', type=float, default=0.0, 
                               help="focal Loss gamma")
    group_decoder.add_argument('--sl_epsilon', type=float, default=0.0, 
                               help="label smoothing loss epsilon")
    
    # Sequence tagging
    group_decoder.add_argument('--scheme', type=str, default='BIOES', 
                               help="sequence tagging scheme", choices=['BIOES', 'BIO2'])
    group_decoder.add_argument('--no_crf', dest='use_crf', default=True, action='store_false', 
                               help="whether to use CRF")
    
    # Span-based
    group_decoder.add_argument('--agg_mode', type=str, default='max_pooling', 
                               help="aggregating mode")
    group_decoder.add_argument('--max_span_size', type=int, default=10, 
                               help="maximum span size")
    group_decoder.add_argument('--max_size_id', type=int, default=9, 
                               help="maximum span size ID")
    group_decoder.add_argument('--size_emb_dim', type=int, default=25, 
                               help="span size embedding dim")
    group_decoder.add_argument('--label_emb_dim', type=int, default=25, 
                               help="chunk label embedding dim")
    
    # Boundary selection
    group_decoder.add_argument('--red_arch', type=str, default='FFN', 
                               help="pre-affine dimension reduction architecture")
    group_decoder.add_argument('--red_dim', type=int, default=150, 
                               help="pre-affine dimension reduction hidden dim")
    group_decoder.add_argument('--red_num_layers', type=int, default=1, 
                               help="number of layers in pre-affine dimension reduction")
    group_decoder.add_argument('--neg_sampling_rate', type=float, default=1.0, 
                               help="negative sampling rate")
    group_decoder.add_argument('--neg_sampling_power_decay', type=float, default=0.0, 
                               help="negative sampling rate power decay parameter")
    group_decoder.add_argument('--neg_sampling_surr_rate', type=float, default=0.0, 
                               help="extra negative sampling rate surrounding positive samples")
    group_decoder.add_argument('--neg_sampling_surr_size', type=int, default=5, 
                               help="extra negative sampling rate surrounding size")
    
    group_decoder.add_argument('--sb_epsilon', type=float, default=0.0, 
                               help="boundary smoothing loss epsilon")
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
                                                            neg_sampling_rate=args.neg_sampling_rate, 
                                                            neg_sampling_power_decay=args.neg_sampling_power_decay, 
                                                            neg_sampling_surr_rate=args.neg_sampling_surr_rate, 
                                                            neg_sampling_surr_size=args.neg_sampling_surr_size, 
                                                            max_span_size=args.max_span_size, 
                                                            size_emb_dim=args.size_emb_dim, 
                                                            in_drop_rates=drop_rates)
    elif args.ck_decoder == 'boundary_selection':
        reduction_config = EncoderConfig(arch=args.red_arch, hid_dim=args.red_dim, num_layers=args.red_num_layers, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0)
        ck_decoder_config = BoundarySelectionDecoderConfig(reduction=reduction_config, 
                                                           fl_gamma=args.fl_gamma,
                                                           sl_epsilon=args.sl_epsilon, 
                                                           neg_sampling_rate=args.neg_sampling_rate, 
                                                           neg_sampling_power_decay=args.neg_sampling_power_decay, 
                                                           neg_sampling_surr_rate=args.neg_sampling_surr_rate, 
                                                           neg_sampling_surr_size=args.neg_sampling_surr_size, 
                                                           sb_epsilon=args.sb_epsilon,
                                                           size_emb_dim=args.size_emb_dim)
    
    if args.attr_decoder == 'None':
        attr_decoder_config = None
    elif args.attr_decoder == 'span_attr_classification':
        attr_decoder_config = SpanAttrClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                                  neg_sampling_rate=args.neg_sampling_rate, 
                                                                  max_size_id=args.max_size_id, 
                                                                  size_emb_dim=args.size_emb_dim, 
                                                                  label_emb_dim=args.label_emb_dim, 
                                                                  in_drop_rates=drop_rates)
    
    if args.rel_decoder == 'None':
        rel_decoder_config = None
    elif args.rel_decoder == 'span_rel_classification':
        rel_decoder_config = SpanRelClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                                fl_gamma=args.fl_gamma,
                                                                sl_epsilon=args.sl_epsilon, 
                                                                neg_sampling_rate=args.neg_sampling_rate, 
                                                                max_size_id=args.max_size_id, 
                                                                size_emb_dim=args.size_emb_dim, 
                                                                label_emb_dim=args.label_emb_dim, 
                                                                in_drop_rates=drop_rates)
    
    decoder_config = JointExtractionDecoderConfig(ck_decoder=ck_decoder_config, 
                                                  attr_decoder=attr_decoder_config,
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
    evaluate_joint_extraction(trainer, dev_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_joint_extraction(trainer, test_set, has_attr=(args.attr_decoder!='None'), has_rel=(args.rel_decoder!='None'), batch_size=args.batch_size, save_preds=args.save_preds)
    if args.save_preds:
        torch.save(test_data, f"{save_path}/test-data-with-preds.pth")
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
