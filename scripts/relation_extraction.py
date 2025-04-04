# -*- coding: utf-8 -*-
from collections import Counter
import os
import sys
import argparse
import datetime
import pdb
import glob
import logging
import pprint
import copy
import torch

from eznlp import auto_device
from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig
from eznlp.model import SpanRelClassificationDecoderConfig, SpecificSpanRelClsDecoderConfig, UnfilteredSpecificSpanRelClsDecoderConfig, MaskedSpanRelClsDecoderConfig
from eznlp.model import ExtractorConfig, SpecificSpanExtractorConfig, MaskedSpanExtractorConfig
from eznlp.model import BertLikePostProcessor
from eznlp.training import Trainer, count_params, evaluate_relation_extraction
from eznlp.training.evaluation import _eval_ent
from eznlp.utils.relation import detect_missing_symmetric

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, build_trainer, header_format
from entity_recognition import collect_IE_assembly_config, process_IE_data


def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)

    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='conll2004',
                            help="dataset name")
    group_data.add_argument('--doc_level', default=False, action='store_true',
                            help="whether to load data at document level")
    group_data.add_argument('--pre_subtokenize', default=False, action='store_true',
                            help="whether to pre-subtokenize words in data")
    group_data.add_argument('--pre_merge_enchars', default=False, action='store_true',
                            help="whether to pre-merge English characters in data")
    group_data.add_argument('--train_chunks_pred_path', type=str, default="",
                            help="path to load predicted (jackknifed) chunks for train split")
    group_data.add_argument('--test_chunks_pred_path', type=str, default="",
                            help="path to load predicted chunks for dev/test splits")
    group_data.add_argument('--no_save_preds', dest='save_preds', default=True, action='store_false',
                            help="whether to save predictions on the dev/test splits")

    group_model = parser.add_argument_group('additional model configurations')
    group_model.add_argument('--bert_load_from_ck_path', type=str, default="",
                             help="path to load BERT weights from a trained ER model")

    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--rel_decoder', type=str, default='span_rel_classification',
                               help="relation decoding method", choices=['span_rel_classification', 'specific_span_rel', 'unfiltered_specific_span_rel', 'masked_span_rel'])

    # Loss
    group_decoder.add_argument('--fl_gamma', type=float, default=0.0,
                               help="focal Loss gamma")
    group_decoder.add_argument('--sl_epsilon', type=float, default=0.0,
                               help="label smoothing loss epsilon")

    # Span-based
    group_decoder.add_argument('--no_context', dest='use_context', default=True, action='store_false',
                               help="whether to use context")
    group_decoder.add_argument('--context_mode', type=str, default='specific',
                               help="context mode")
    group_decoder.add_argument('--context_ext_win', type=int, default=0,
                               help="window size of extended context")
    group_decoder.add_argument('--context_no_exc_ck', dest='context_exc_ck', default=True, action='store_false',
                               help="whether to exclude chunk ranges for context")
    group_decoder.add_argument('--agg_mode', type=str, default='max_pooling',
                               help="aggregating mode")
    group_decoder.add_argument('--size_emb_dim', type=int, default=0,
                               help="span size embedding dim")
    group_decoder.add_argument('--label_emb_dim', type=int, default=0,
                               help="chunk label embedding dim")
    group_decoder.add_argument('--fusing_mode', type=str, default='concat',
                               help="fusing mode for span/context representations")
    group_decoder.add_argument('--ck_loss_weight', type=float, default=0.0,
                               help="weight for entity loss")
    group_decoder.add_argument('--l2_loss_weight', type=float, default=0.0,
                               help="weight for l2-regularization on affine-fusor")
    group_decoder.add_argument('--use_inv_rel', default=False, action='store_true',
                               help="whether to use inverse relation for bidirectional prediction")
    group_decoder.add_argument('--comp_sym_rel', default=False, action='store_true',
                               help="whether to complement symmetric relations")
    group_decoder.add_argument('--check_rht_labels', default=False, action='store_true',
                               help="whether to check the combination of triplet labels")

    # Boundary selection (*)
    group_decoder.add_argument('--red_arch', type=str, default='FFN',
                               help="pre-affine dimension reduction architecture")
    group_decoder.add_argument('--red_dim', type=int, default=150,
                               help="pre-affine dimension reduction hidden dim")
    group_decoder.add_argument('--red_num_layers', type=int, default=1,
                               help="number of layers in pre-affine dimension reduction")
    group_decoder.add_argument('--neg_sampling_rate', type=float, default=1.0,
                               help="negative sampling rate")
    group_decoder.add_argument('--ss_epsilon', type=float, default=0.0,
                               help="simplified smoothing loss epsilon")

    # Specific span classification: Span-specific encoder (SSE)
    group_decoder.add_argument('--sse_no_share_weights_ext', dest='sse_share_weights_ext', default=True, action='store_false',
                               help="whether to share weights between span-bert and bert encoders")
    group_decoder.add_argument('--sse_no_share_weights_int', dest='sse_share_weights_int', default=True, action='store_false',
                               help="whether to share weights across span-bert encoders")
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
    group_decoder.add_argument('--sse_use_init_dist_emb', default=False, action='store_true',
                               help="whether to use initial pair distance embeddings")
    return parse_to_args(parser)



def build_RE_config(args: argparse.Namespace):
    (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    reduction_config = EncoderConfig(arch=args.red_arch, hid_dim=args.red_dim, num_layers=args.red_num_layers, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0)

    if args.rel_decoder == 'span_rel_classification':
        decoder_config = SpanRelClassificationDecoderConfig(use_context=args.use_context,
                                                            size_emb_dim=args.size_emb_dim,
                                                            label_emb_dim=args.label_emb_dim,
                                                            fusing_mode=args.fusing_mode,
                                                            reduction=reduction_config,
                                                            agg_mode=args.agg_mode,
                                                            ck_loss_weight=args.ck_loss_weight,
                                                            l2_loss_weight=args.l2_loss_weight,
                                                            comp_sym_rel=args.comp_sym_rel,
                                                            use_inv_rel=args.use_inv_rel,
                                                            check_rht_labels=args.check_rht_labels,
                                                            ss_epsilon=args.ss_epsilon,
                                                            fl_gamma=args.fl_gamma,
                                                            sl_epsilon=args.sl_epsilon,
                                                            neg_sampling_rate=args.neg_sampling_rate)
    elif args.rel_decoder == 'specific_span_rel':
        decoder_config = SpecificSpanRelClsDecoderConfig(use_context=args.use_context,
                                                         context_mode=args.context_mode,
                                                         size_emb_dim=args.size_emb_dim,
                                                         label_emb_dim=args.label_emb_dim,
                                                         fusing_mode=args.fusing_mode,
                                                         reduction=reduction_config,
                                                         agg_mode=args.agg_mode,
                                                         ck_loss_weight=args.ck_loss_weight,
                                                         l2_loss_weight=args.l2_loss_weight,
                                                         comp_sym_rel=args.comp_sym_rel,
                                                         use_inv_rel=args.use_inv_rel,
                                                         check_rht_labels=args.check_rht_labels,
                                                         ss_epsilon=args.ss_epsilon,
                                                         fl_gamma=args.fl_gamma,
                                                         sl_epsilon=args.sl_epsilon,
                                                         neg_sampling_rate=args.neg_sampling_rate,
                                                         min_span_size=args.sse_min_span_size,
                                                         max_span_size_cov_rate=args.sse_max_span_size_cov_rate,
                                                         max_span_size=(args.sse_max_span_size if args.sse_max_span_size>0 else None))
    elif args.rel_decoder == 'unfiltered_specific_span_rel':
        decoder_config = UnfilteredSpecificSpanRelClsDecoderConfig(fusing_mode=args.fusing_mode,
                                                                   reduction=reduction_config,
                                                                   fl_gamma=args.fl_gamma,
                                                                   sl_epsilon=args.sl_epsilon,
                                                                   neg_sampling_rate=args.neg_sampling_rate,
                                                                   min_span_size=args.sse_min_span_size,
                                                                   max_span_size_cov_rate=args.sse_max_span_size_cov_rate,
                                                                   max_span_size=(args.sse_max_span_size if args.sse_max_span_size>0 else None))
    elif args.rel_decoder == 'masked_span_rel':
        decoder_config = MaskedSpanRelClsDecoderConfig(use_context=args.use_context,
                                                       context_mode=args.context_mode,
                                                       context_ext_win=args.context_ext_win,
                                                       context_exc_ck=args.context_exc_ck,
                                                       size_emb_dim=args.size_emb_dim,
                                                       label_emb_dim=args.label_emb_dim,
                                                       fusing_mode=args.fusing_mode,
                                                       reduction=reduction_config,
                                                       ck_loss_weight=args.ck_loss_weight,
                                                       l2_loss_weight=args.l2_loss_weight,
                                                       comp_sym_rel=args.comp_sym_rel,
                                                       use_inv_rel=args.use_inv_rel,
                                                       check_rht_labels=args.check_rht_labels,
                                                       ss_epsilon=args.ss_epsilon,
                                                       fl_gamma=args.fl_gamma,
                                                       sl_epsilon=args.sl_epsilon,
                                                       neg_sampling_rate=args.neg_sampling_rate)

    if args.rel_decoder.startswith('specific_span'):
        return SpecificSpanExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)
    elif args.rel_decoder.startswith('masked_span'):
        return MaskedSpanExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)
    else:
        return ExtractorConfig(**collect_IE_assembly_config(args), decoder=decoder_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
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

    train_data, dev_data, test_data = load_data(args)
    args.language = dataset2language[args.dataset]

    if args.train_with_dev:
        train_data = train_data + dev_data
        dev_data   = []

    if len(args.train_chunks_pred_path) > 0:
        logger.info("Jackknifed chunks are provided, using merged chunks for training")
        train_data_pred = torch.load(f"{args.train_chunks_pred_path}/train.data.pred.pth")
        logger.info("Evaluating jackknifed chunks on train-set")
        set_chunks_pred = [entry['chunks_pred'] for entry in train_data_pred]
        set_chunks_gold = [entry['chunks'] for entry in train_data]
        _eval_ent(set_chunks_gold, set_chunks_pred)
        for entry, entry_pred in zip(train_data, train_data_pred):
            entry['chunks_pred'] = entry_pred['chunks_pred']
    else:
        logger.info("Jackknifed chunks are unavailable, using gold chunks for training")
        for entry in train_data:
            entry['chunks_pred'] = entry['chunks']

    if len(args.test_chunks_pred_path) > 0:
        logger.info("Predicted chunks are provided, performing end-to-end evaluation")
        dev_data_pred = torch.load(f"{args.test_chunks_pred_path}/dev.data.pred.pth")
        logger.info("Evaluating predicted chunks on dev-set")
        set_chunks_pred = [entry['chunks_pred'] for entry in dev_data_pred]
        set_chunks_gold = [entry['chunks'] for entry in dev_data]
        _eval_ent(set_chunks_gold, set_chunks_pred)
        for entry, entry_pred in zip(dev_data, dev_data_pred):
            entry['chunks_pred'] = entry_pred['chunks_pred']

        test_data_pred = torch.load(f"{args.test_chunks_pred_path}/test.data.pred.pth")
        logger.info("Evaluating predicted chunks on test-set")
        set_chunks_pred = [entry['chunks_pred'] for entry in test_data_pred]
        set_chunks_gold = [entry['chunks'] for entry in test_data]
        _eval_ent(set_chunks_gold, set_chunks_pred)
        for entry, entry_pred in zip(test_data, test_data_pred):
            entry['chunks_pred'] = entry_pred['chunks_pred']

    else:
        logger.warning("Predicted chunks are unavailable, using gold chunks for evaluation")
        for entry in dev_data+test_data:
            entry['chunks_pred'] = entry['chunks']

    config = build_RE_config(args)
    train_data, dev_data, test_data = process_IE_data(train_data, dev_data, test_data, args, config)

    # Remove entries without chunks, because either gold or predicted relations must be empty
    train_data = [entry for entry in train_data if len(entry['chunks'])+len(entry['chunks_pred']) > 0]
    dev_data   = [entry for entry in dev_data   if len(entry['chunks'])+len(entry['chunks_pred']) > 0]
    test_data  = [entry for entry in test_data  if len(entry['chunks'])+len(entry['chunks_pred']) > 0]

    # Add missing symmetric relations
    if args.dataset.startswith('ace2004_rel_cv') or args.dataset in ('ace2005_rel', 'SciERC'):
        if args.dataset.startswith('ace'):
            config.decoder.sym_rel_labels = ['PER-SOC']
        elif args.dataset == 'SciERC':
            config.decoder.sym_rel_labels = ['COMPARE', 'CONJUNCTION']

        for data in [train_data, dev_data, test_data]:
            counter = Counter()
            for entry in data:
                missing_relations = detect_missing_symmetric(entry['relations'], config.decoder.sym_rel_labels)
                counter.update([rel[0] for rel in missing_relations])
                entry['relations'].extend(missing_relations)
            logger.info("Adding missing symmetric relations: " + " | ".join(f"{label}: {c}" for label, c in counter.items()))


    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data)
    dev_set   = Dataset(dev_data,  train_set.config, training=False)
    test_set  = Dataset(test_data, train_set.config, training=False)
    logger.info(train_set.summary)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, collate_fn=dev_set.collate) if len(dev_data) > 0 else None

    logger.info(header_format("Building", sep='-'))
    if args.dataset.startswith('ace2004_rel_cv'):
        # A relation with exceptionally long distance exists in ACE 2004
        config.decoder.max_cp_dist = 35

    if len(args.bert_load_from_ck_path) > 0:
        if args.bert_load_from_ck_path == 'test_chunks_pred_path':
            args.bert_load_from_ck_path = args.test_chunks_pred_path

        # Load the BERT weights from the ER model
        ck_model_file_path = glob.glob(f"{args.bert_load_from_ck_path}/*-config.pth")[0].replace("-config", "")
        ck_model = torch.load(ck_model_file_path, map_location='cpu')
        bert_like = copy.deepcopy(ck_model.bert_like.bert_like)
        config.bert_like.bert_like = bert_like
        config.masked_span_bert_like.bert_like = bert_like
        model = config.instantiate()

        # Load some other weights from the ER model
        assert type(model.masked_span_bert_like.init_aggregating) == type(ck_model.span_bert_like.init_aggregating)
        model.masked_span_bert_like.init_aggregating = copy.deepcopy(ck_model.span_bert_like.init_aggregating)
        assert model.masked_span_bert_like.size_embedding.num_embeddings == ck_model.span_bert_like.size_embedding.num_embeddings
        model.masked_span_bert_like.size_embedding = copy.deepcopy(ck_model.span_bert_like.size_embedding)
        model.to(device)
    else:
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
    evaluate_relation_extraction(trainer, dev_set, batch_size=args.batch_size)
    logger.info("Evaluating on test-set")
    evaluate_relation_extraction(trainer, test_set, batch_size=args.batch_size)

    if args.save_preds:
        postprocessor = BertLikePostProcessor(verbose=args.log_terminal)

        logger.info("Saving predictions on dev-set")
        set_relations_pred = trainer.predict(dev_set, batch_size=args.batch_size)
        for entry, relations_pred in zip(dev_data, set_relations_pred):
            entry['relations_pred'] = relations_pred
        dev_data_pred = postprocessor.restore_for_data(dev_data)
        torch.save(dev_data_pred, f"{save_path}/dev.data.pred.pth")

        logger.info("Saving predictions on test-set")
        set_relations_pred = trainer.predict(test_set, batch_size=args.batch_size)
        for entry, relations_pred in zip(test_data, set_relations_pred):
            entry['relations_pred'] = relations_pred
        test_data_pred = postprocessor.restore_for_data(test_data)
        torch.save(test_data_pred, f"{save_path}/test.data.pred.pth")

    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
