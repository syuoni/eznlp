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
from eznlp.token import LexiconTokenizer
from eznlp.dataset import Dataset
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig, CharConfig, SoftLexiconConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model import SequenceTaggingDecoderConfig, SpanClassificationDecoderConfig, BoundarySelectionDecoderConfig
from eznlp.model import ExtractorConfig
from eznlp.model.bert_like import segment_uniformly_for_bert_like
from eznlp.training import Trainer, count_params, evaluate_entity_recognition

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_pretrained, load_vectors, build_trainer, header_format, profile


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2003', 
                            help="dataset name")
    group_data.add_argument('--doc_level', default=False, action='store_true', 
                            help="whether to load data at document level")
    group_data.add_argument('--pipeline', default=False, action='store_true', 
                            help="whether to save predicted chunks for pipeline")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging', 
                               help="chunk decoding method", choices=['sequence_tagging', 'span_classification', 'boundary_selection'])
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
    
    # Boundary selection
    group_decoder.add_argument('--no_biaffine', dest='use_biaffine', default=True, action='store_false', 
                               help="whether to use biaffine")
    group_decoder.add_argument('--affine_arch', type=str, default='FFN', 
                               help="affine encoder architecture")
    group_decoder.add_argument('--neg_sampling_rate', type=float, default=1.0, 
                               help="Negative sampling rate")
    group_decoder.add_argument('--hard_neg_sampling_rate', type=float, default=1.0, 
                               help="Hard negative sampling rate")
    group_decoder.add_argument('--hard_neg_sampling_size', type=int, default=5, 
                               help="Hard negative sampling window size")
    group_decoder.add_argument('--sb_epsilon', type=float, default=0.0, 
                               help="Boundary smoothing loss epsilon")
    group_decoder.add_argument('--sb_size', type=int, default=1, 
                               help="Boundary smoothing window size")
    return parse_to_args(parser)


def collect_IE_assembly_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    ohots_config = {}
    if args.emb_dim > 0:
        vectors = load_vectors(args.language, args.emb_dim, unigram=True)
        ohots_config['text'] = OneHotConfig(field='text', vectors=vectors, emb_dim=args.emb_dim, freeze=args.emb_freeze)
        
    if args.language.lower() == 'chinese' and args.use_bigram:
        vectors = load_vectors(args.language, 50, bigram=True)
        ohots_config['bigram'] = OneHotConfig(field='bigram', vectors=vectors, emb_dim=50, freeze=args.emb_freeze)
    
    ohots_config = ConfigDict(ohots_config) if len(ohots_config) > 0 else None
    
    
    if args.language.lower() == 'chinese' and args.use_softword:
        mhots_config = ConfigDict({'softword': MultiHotConfig(field='softword', use_emb=False)})
    else:
        mhots_config = None
    
    if args.language.lower() == 'english' and args.char_arch.lower() != 'none':
        char_config = CharConfig(emb_dim=16,  
                                 encoder=EncoderConfig(arch=args.char_arch, hid_dim=128, num_layers=1, 
                                                       in_drop_rates=(args.drop_rate, 0.0, 0.0)))
        nested_ohots_config = ConfigDict({'char': char_config})
    elif args.language.lower() == 'chinese' and args.use_softlexicon:
        vectors = load_vectors(args.language, 50)
        nested_ohots_config = ConfigDict({'softlexicon': SoftLexiconConfig(vectors=vectors, emb_dim=50, freeze=args.emb_freeze)})
    else:
        nested_ohots_config = None
    
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
        # Cased tokenizer for NER task
        bert_like, tokenizer = load_pretrained(args.bert_arch, args, cased=True)
        bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_like, arch=args.bert_arch, 
                                          freeze=False, use_truecase='cased' in os.path.basename(bert_like.name_or_path).split('-'))
    else:
        bert_like_config = None
    
    return {'ohots': ohots_config, 
            'mhots': mhots_config, 
            'nested_ohots': nested_ohots_config, 
            'intermediate1': interm1_config, 
            'elmo': elmo_config, 
            'flair_fw': flair_fw_config, 
            'flair_bw': flair_bw_config, 
            'bert_like': bert_like_config, 
            'intermediate2': interm2_config}


def build_ER_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    if args.ck_decoder == 'sequence_tagging':
        decoder_config = SequenceTaggingDecoderConfig(scheme=args.scheme, 
                                                      use_crf=args.use_crf, 
                                                      fl_gamma=args.fl_gamma,
                                                      sl_epsilon=args.sl_epsilon, 
                                                      in_drop_rates=drop_rates)
    elif args.ck_decoder == 'span_classification':
        decoder_config = SpanClassificationDecoderConfig(agg_mode=args.agg_mode, 
                                                         fl_gamma=args.fl_gamma,
                                                         sl_epsilon=args.sl_epsilon, 
                                                         num_neg_chunks=args.num_neg_chunks, 
                                                         max_span_size=args.max_span_size, 
                                                         size_emb_dim=args.ck_size_emb_dim, 
                                                         in_drop_rates=drop_rates)
    elif args.ck_decoder == 'boundary_selection':
        decoder_config = BoundarySelectionDecoderConfig(use_biaffine=args.use_biaffine, 
                                                        affine=EncoderConfig(arch=args.affine_arch, hid_dim=150, num_layers=1, in_drop_rates=(0.4, 0.0, 0.0), hid_drop_rate=0.2), 
                                                        fl_gamma=args.fl_gamma,
                                                        sl_epsilon=args.sl_epsilon, 
                                                        neg_sampling_rate=args.neg_sampling_rate, 
                                                        hard_neg_sampling_rate=args.hard_neg_sampling_rate, 
                                                        hard_neg_sampling_size=args.hard_neg_sampling_size, 
                                                        sb_epsilon=args.sb_epsilon, 
                                                        sb_size=args.sb_size,
                                                        hid_drop_rates=drop_rates)
    return ExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)


def process_IE_data(train_data, dev_data, test_data, args, config):
    if (config.bert_like is not None and 
            ((args.dataset in ('SIGHAN2006', 'yidu_s4k')) or 
             (args.dataset in ('conll2003', 'conll2012') and getattr(args, 'doc_level', False)))):
        train_data = segment_uniformly_for_bert_like(train_data, config.bert_like.tokenizer, verbose=args.log_terminal)
        dev_data   = segment_uniformly_for_bert_like(dev_data,   config.bert_like.tokenizer, verbose=args.log_terminal)
        test_data  = segment_uniformly_for_bert_like(test_data,  config.bert_like.tokenizer, verbose=args.log_terminal)
    
    if getattr(args, 'use_softword', False) or getattr(args, 'use_softlexicon', False):
        if config.nested_ohots is not None and 'softlexicon' in config.nested_ohots.keys():
            vectors = config.nested_ohots['softlexicon'].vectors
        else:
            vectors = load_vectors(args.language, 50)
        tokenizer = LexiconTokenizer(vectors.itos)
        for data in [train_data, dev_data, test_data]:
            for data_entry in data:
                data_entry['tokens'].build_softwords(tokenizer.tokenize)
                data_entry['tokens'].build_softlexicons(tokenizer.tokenize)
    
    return train_data, dev_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-ER/{timestamp}"
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
    config = build_ER_config(args)
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
    
    if args.profile:
        prof = profile(trainer, train_loader)
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
    evaluate_entity_recognition(trainer, dev_set, batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_entity_recognition(trainer, test_set, batch_size=args.batch_size)
    
    
    if args.pipeline:
        # Replace gold chunks with predicted chunks for pipeline
        if args.train_with_dev:
            # Retrieve the original splits
            train_set = Dataset(train_data, train_set.config, training=True)
            dev_set   = Dataset(dev_data,   train_set.config, training=False)
        
        train_set_chunks_pred = trainer.predict(train_set, batch_size=args.batch_size)
        for ex, chunks_pred in zip(train_data, train_set_chunks_pred):
            ex['chunks'] = ex['chunks'] + [ck for ck in chunks_pred if ck not in ex['chunks']]
        
        dev_set_chunks_pred = trainer.predict(dev_set, batch_size=args.batch_size)
        for ex, chunks_pred in zip(dev_data, dev_set_chunks_pred):
            ex['chunks'] = chunks_pred
        
        test_set_chunks_pred = trainer.predict(test_set, batch_size=args.batch_size)
        for ex, chunks_pred in zip(test_data, test_set_chunks_pred):
            ex['chunks'] = chunks_pred
            
        torch.save((train_data, dev_data, test_data), f"{save_path}/data-with-chunks-pred.pth")
    
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
