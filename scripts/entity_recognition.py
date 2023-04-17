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
from eznlp.utils.chunk import FLAT, NESTED, detect_nested, _is_ordered_nested
from eznlp.nn.init import reinit_bert_like_
from eznlp.dataset import Dataset
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig, CharConfig, SoftLexiconConfig
from eznlp.model import ELMoConfig, BertLikeConfig, SpanBertLikeConfig, MaskedSpanBertLikeConfig, FlairConfig
from eznlp.model import SequenceTaggingDecoderConfig, SpanClassificationDecoderConfig, BoundarySelectionDecoderConfig, SpecificSpanClsDecoderConfig
from eznlp.model import ExtractorConfig, SpecificSpanExtractorConfig
from eznlp.model import BertLikePreProcessor, BertLikePostProcessor
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
    group_data.add_argument('--pre_truecase', default=False, action='store_true', 
                            help="whether to pre-truecase data")
    group_data.add_argument('--pre_subtokenize', default=False, action='store_true', 
                            help="whether to pre-subtokenize words in data")
    group_data.add_argument('--pre_merge_enchars', default=False, action='store_true', 
                            help="whether to pre-merge English characters in data")
    group_data.add_argument('--corrupt_rate', type=float, default=0.0, 
                            help="boundary corrupt rate")
    group_data.add_argument('--remove_nested', default=False, action='store_true', 
                            help="whether to remove nested entities in the train/dev splits")
    group_data.add_argument('--eval_inex', default=False, action='store_true', 
                            help="whether to evaluate internal/external-entity NER results")
    group_data.add_argument('--save_preds', default=False, action='store_true', 
                            help="whether to save predictions on the dev/test splits (e.g., in case without ground truth)")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--ck_decoder', type=str, default='sequence_tagging', 
                               help="chunk decoding method", choices=['sequence_tagging', 'span_classification', 'boundary_selection', 'specific_span'])
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
    group_decoder.add_argument('--size_emb_dim', type=int, default=25, 
                               help="span size embedding dim")
    group_decoder.add_argument('--inex_mkmmd_lambda', type=float, default=0.0, 
                               help="weight of internal/external-entity MK-MMD loss")
    group_decoder.add_argument('--inex_chunk_priority_training', type=str, default='confidence', 
                               help="chunk priority in the training phase")
    group_decoder.add_argument('--inex_chunk_priority_testing', type=str, default='confidence', 
                               help="chunk priority in the testing phase")
    
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
    group_decoder.add_argument('--nested_sampling_rate', type=float, default=1.0, 
                               help="sampling rate for spans nested in positive samples")
    
    group_decoder.add_argument('--sb_epsilon', type=float, default=0.0, 
                               help="boundary smoothing loss epsilon")
    group_decoder.add_argument('--sb_size', type=int, default=1, 
                               help="boundary smoothing window size")
    group_decoder.add_argument('--sb_adj_factor', type=float, default=1.0, 
                               help="boundary smoothing probability adjust factor")
    
    # Specific span classification: Span-specific encoder (SSE)
    group_decoder.add_argument('--sse_no_share_weights_ext', dest='sse_share_weights_ext', default=True, action='store_false', 
                               help="whether to share weights between span-bert and bert encoders")
    group_decoder.add_argument('--sse_no_share_weights_int', dest='sse_share_weights_int', default=True, action='store_false', 
                               help="whether to share weights across span-bert encoders")
    group_decoder.add_argument('--sse_no_share_interm2', dest='sse_share_interm2', default=True, action='store_false', 
                               help="whether to share interm2 between span-bert and bert encoders")
    group_decoder.add_argument('--sse_init_agg_mode', type=str, default='max_pooling', 
                               help="initial aggregating mode for span-bert enocder")
    group_decoder.add_argument('--sse_init_drop_rate', type=float, default=0.2, 
                               help="dropout rate before initial aggregating")
    group_decoder.add_argument('--sse_num_layers', type=int, default=-1, 
                               help="number of span-bert encoder layers (negative values are set to `None`)")
    group_decoder.add_argument('--sse_min_span_size', type=int, default=2, 
                               help="minimum span size", choices=[2, 1])
    group_decoder.add_argument('--sse_max_span_size_cov_rate', type=float, default=0.995, 
                               help="coverage rate of maximum span size")
    group_decoder.add_argument('--sse_max_span_size', type=int, default=-1, 
                               help="maximum span size (negative values are set to `None`)")
    group_decoder.add_argument('--sse_use_init_size_emb', default=False, action='store_true', 
                               help="whether to use initial span size embeddings")
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
        bert_like, tokenizer = load_pretrained(args.bert_arch, args)
        if args.bert_reinit:
            reinit_bert_like_(bert_like)
        bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert_like, arch=args.bert_arch, from_subtokenized=args.pre_subtokenize, freeze=args.bert_freeze)
        
        if getattr(args, 'ck_decoder', '').startswith('specific_span') or getattr(args, 'rel_decoder', '').startswith('specific_span'):
            bert_like_config.output_hidden_states = True
            span_bert_like_config = SpanBertLikeConfig(bert_like=bert_like, arch=args.bert_arch, freeze=args.bert_freeze, 
                                                       num_layers=None if args.sse_num_layers < 0 else args.sse_num_layers, 
                                                       use_init_size_emb=args.sse_use_init_size_emb, 
                                                       share_weights_ext=args.sse_share_weights_ext, 
                                                       share_weights_int=args.sse_share_weights_int, 
                                                       init_agg_mode=args.sse_init_agg_mode, 
                                                       init_drop_rate=args.sse_init_drop_rate)
        else:
            span_bert_like_config = None
        
        if getattr(args, 'ck_decoder', '').startswith('masked_span') or getattr(args, 'rel_decoder', '').startswith('masked_span'):
            bert_like_config.output_hidden_states = True
            masked_span_bert_like_config = MaskedSpanBertLikeConfig(bert_like=bert_like, arch=args.bert_arch, freeze=args.bert_freeze, 
                                                                    num_layers=None if args.sse_num_layers < 0 else args.sse_num_layers, 
                                                                    use_init_size_emb=args.sse_use_init_size_emb, 
                                                                    use_init_dist_emb=args.sse_use_init_dist_emb, 
                                                                    share_weights_ext=args.sse_share_weights_ext, 
                                                                    share_weights_int=args.sse_share_weights_int, 
                                                                    init_agg_mode=args.sse_init_agg_mode, 
                                                                    init_drop_rate=args.sse_init_drop_rate)
        else:
            masked_span_bert_like_config = None
        
    else:
        bert_like_config = None
        span_bert_like_config = None
        masked_span_bert_like_config = None
    
    return {'ohots': ohots_config, 
            'mhots': mhots_config, 
            'nested_ohots': nested_ohots_config, 
            'intermediate1': interm1_config, 
            'elmo': elmo_config, 
            'flair_fw': flair_fw_config, 
            'flair_bw': flair_bw_config, 
            'bert_like': bert_like_config, 
            'span_bert_like': span_bert_like_config, 
            'masked_span_bert_like': masked_span_bert_like_config, 
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
                                                         neg_sampling_rate=args.neg_sampling_rate, 
                                                         neg_sampling_power_decay=args.neg_sampling_power_decay, 
                                                         neg_sampling_surr_rate=args.neg_sampling_surr_rate, 
                                                         neg_sampling_surr_size=args.neg_sampling_surr_size, 
                                                         nested_sampling_rate=args.nested_sampling_rate, 
                                                         sb_epsilon=args.sb_epsilon, 
                                                         sb_size=args.sb_size, 
                                                         sb_adj_factor=args.sb_adj_factor, 
                                                         max_span_size=args.max_span_size, 
                                                         size_emb_dim=args.size_emb_dim, 
                                                         inex_mkmmd_lambda=args.inex_mkmmd_lambda, 
                                                         in_drop_rates=drop_rates)
    elif args.ck_decoder == 'boundary_selection':
        reduction_config = EncoderConfig(arch=args.red_arch, hid_dim=args.red_dim, num_layers=args.red_num_layers, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0)
        decoder_config = BoundarySelectionDecoderConfig(reduction=reduction_config, 
                                                        fl_gamma=args.fl_gamma,
                                                        sl_epsilon=args.sl_epsilon, 
                                                        neg_sampling_rate=args.neg_sampling_rate, 
                                                        neg_sampling_power_decay=args.neg_sampling_power_decay, 
                                                        neg_sampling_surr_rate=args.neg_sampling_surr_rate, 
                                                        neg_sampling_surr_size=args.neg_sampling_surr_size, 
                                                        nested_sampling_rate=args.nested_sampling_rate, 
                                                        sb_epsilon=args.sb_epsilon, 
                                                        sb_size=args.sb_size,
                                                        sb_adj_factor=args.sb_adj_factor, 
                                                        size_emb_dim=args.size_emb_dim)
    elif args.ck_decoder == 'specific_span':
        decoder_config = SpecificSpanClsDecoderConfig(fl_gamma=args.fl_gamma,
                                                      sl_epsilon=args.sl_epsilon, 
                                                      neg_sampling_rate=args.neg_sampling_rate, 
                                                      neg_sampling_power_decay=args.neg_sampling_power_decay, 
                                                      neg_sampling_surr_rate=args.neg_sampling_surr_rate, 
                                                      neg_sampling_surr_size=args.neg_sampling_surr_size, 
                                                      nested_sampling_rate=args.nested_sampling_rate, 
                                                      sb_epsilon=args.sb_epsilon, 
                                                      sb_size=args.sb_size,
                                                      sb_adj_factor=args.sb_adj_factor, 
                                                      min_span_size=args.sse_min_span_size, 
                                                      max_span_size_cov_rate=args.sse_max_span_size_cov_rate, 
                                                      max_span_size=(args.sse_max_span_size if args.sse_max_span_size>0 else None), 
                                                      size_emb_dim=args.size_emb_dim, 
                                                      inex_mkmmd_lambda=args.inex_mkmmd_lambda, 
                                                      in_drop_rates=drop_rates)
    
    if args.ck_decoder == 'specific_span':
        return SpecificSpanExtractorConfig(**collect_IE_assembly_config(args), share_interm2=args.sse_share_interm2, decoder=decoder_config)
    else:
        return ExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)


def process_IE_data(train_data, dev_data, test_data, args, config):
    if config.bert_like is not None: 
        preprocessor = BertLikePreProcessor(config.bert_like.tokenizer, model_max_length=args.bert_max_length, verbose=args.log_terminal)
        
        if getattr(args, 'pre_truecase', False):
            assert not getattr(config.bert_like.tokenizer, 'do_lower_case', False)
            train_data = preprocessor.truecase_for_data(train_data, )
            dev_data   = preprocessor.truecase_for_data(dev_data)
            test_data  = preprocessor.truecase_for_data(test_data)
        
        if (args.doc_level and (args.dataset in ('conll2003', 'conll2003nff', 'conll2012', 'genia', 'genia_yu2020acl', 'kbp2017', 'SciERC') 
                             or args.dataset.startswith(('ace2004_rel', 'ace2005_rel')))):
            if args.dataset.startswith('conll'):
                doc_key = 'doc_idx'
            elif args.dataset.startswith('genia'):
                doc_key = 'doc_key'
            elif args.dataset.startswith('kbp2017'):
                doc_key = 'org_id'
            elif args.dataset.startswith(('ace2004_rel', 'ace2005_rel', 'SciERC')):
                doc_key = 'doc_key'
            
            train_data = preprocessor.merge_sentences_for_data(train_data, doc_key=doc_key)
            dev_data   = preprocessor.merge_sentences_for_data(dev_data,   doc_key=doc_key)
            test_data  = preprocessor.merge_sentences_for_data(test_data,  doc_key=doc_key)
        
        if args.dataset in ('SIGHAN2006', 'yidu_s4k', 'cmeee'):
            train_data = preprocessor.segment_sentences_for_data(train_data, update_raw_idx=True)
            dev_data   = preprocessor.segment_sentences_for_data(dev_data,   update_raw_idx=True)
            test_data  = preprocessor.segment_sentences_for_data(test_data,  update_raw_idx=True)
        
        if args.pre_subtokenize:
            train_data = preprocessor.subtokenize_for_data(train_data)
            dev_data   = preprocessor.subtokenize_for_data(dev_data)
            test_data  = preprocessor.subtokenize_for_data(test_data)
        
        if args.pre_merge_enchars:
            train_data = preprocessor.merge_enchars_for_data(train_data)
            dev_data   = preprocessor.merge_enchars_for_data(dev_data)
            test_data  = preprocessor.merge_enchars_for_data(test_data)
            if args.dataset in ('WeiboNER', ):
                for entry in train_data:
                    entry['chunks'] = [(label, round(start), round(end)) for label, start, end in entry['chunks']]
    
    if args.use_softword or args.use_softlexicon:
        if config.nested_ohots is not None and 'softlexicon' in config.nested_ohots.keys():
            vectors = config.nested_ohots['softlexicon'].vectors
        else:
            vectors = load_vectors(args.language, 50)
        tokenizer = LexiconTokenizer(vectors.itos)
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                entry['tokens'].build_softwords(tokenizer.tokenize)
                entry['tokens'].build_softlexicons(tokenizer.tokenize)
    
    if getattr(args, 'remove_nested', False):
        logger.info("Removing nested entities...")
        for data in [train_data, dev_data, test_data]:
            for entry in data:
                entry['chunks_nested'] = detect_nested(entry['chunks'])
                entry['chunks'] = list(set(entry['chunks']) - set(entry['chunks_nested']))
    
    return train_data, dev_data, test_data



def conll2003nff_post_process(chunks):
    processed = []
    for ck1 in chunks:
        if ck1[0] == 'ORG' and any(_is_ordered_nested(ck1, ck2) and (ck1 != ck2) for ck2 in chunks):
            processed.append(('LOC', *ck1[1:]))
        elif ck1[0] == 'PER' and any((ck2[0] == 'PER') and _is_ordered_nested(ck1, ck2) and (ck1 != ck2) for ck2 in chunks):
            continue
        else:
            processed.append(ck1)
    return processed


dataset2pp_callback = {"conll2003nff": conll2003nff_post_process}


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
    if args.remove_nested:
        config.decoder.overlapping_level = FLAT
        config.decoder.chunk_priority = args.inex_chunk_priority_training
    
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
    
    # Save the final version
    # torch.save(model, f"{save_path}/{config.name}.fv.pth")
    
    logger.info(header_format("Evaluating", sep='-'))
    model = torch.load(f"{save_path}/{config.name}.pth", map_location=device)
    trainer = Trainer(model, device=device)
    
    if args.train_with_dev:
        # Restore the original splits
        train_set = Dataset(train_data, train_set.config, training=False)
        dev_set   = Dataset(dev_data,   train_set.config, training=False)
    
    if args.remove_nested:
        logger.info("Adding back nested entities...")
        config.decoder.overlapping_level = NESTED
        model.decoder.overlapping_level = NESTED
        config.decoder.chunk_priority = args.inex_chunk_priority_testing
        model.decoder.chunk_priority = args.inex_chunk_priority_testing
        
        for data in [dev_data, test_data]:
            for entry in data:
                entry['chunks'] = entry['chunks'] + entry['chunks_nested']
                del entry['chunks_nested']
        dev_set   = Dataset(dev_data,  train_set.config, training=False)
        test_set  = Dataset(test_data, train_set.config, training=False)
    
    logger.info("Evaluating on dev-set")
    evaluate_entity_recognition(trainer, dev_set,  batch_size=args.batch_size, eval_inex=args.eval_inex)
    logger.info("Evaluating on test-set")
    evaluate_entity_recognition(trainer, test_set, batch_size=args.batch_size, eval_inex=args.eval_inex, pp_callback=dataset2pp_callback.get(args.dataset, None))
    
    if args.save_preds:
        postprocessor = BertLikePostProcessor(verbose=args.log_terminal)
        
        logger.info("Saving predictions on dev-set")
        set_chunks_pred = trainer.predict(dev_set, batch_size=args.batch_size)
        for entry, chunks_pred in zip(dev_data, set_chunks_pred): 
            entry['chunks'] = chunks_pred
        set_chunks_pred = postprocessor.restore_chunks_for_data(dev_data)
        torch.save(set_chunks_pred, f"{save_path}/dev.chunks.pred.pth")
        
        logger.info("Saving predictions on test-set")
        set_chunks_pred = trainer.predict(test_set, batch_size=args.batch_size)
        for entry, chunks_pred in zip(test_data, set_chunks_pred):
            entry['chunks'] = chunks_pred
        set_chunks_pred = postprocessor.restore_chunks_for_data(test_data)
        torch.save(set_chunks_pred, f"{save_path}/test.chunks.pred.pth")
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
