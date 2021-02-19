# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import random
import logging
import pprint
import numpy as np
import torch
import allennlp.modules
import transformers
import flair

from eznlp.config import ConfigDict
from eznlp.model import CharConfig, OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.pretrained import GloVe, ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.training.utils import LRLambda
from eznlp.training.utils import count_params, collect_params, check_param_groups


from utils import load_data, evaluate_sequence_tagging, header_format


SEED = 515
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--debug', dest='debug', default=False, action='store_true', 
                        help="whether to use pdb for debug")
    
    parser.add_argument('--device', type=str, default='cpu', 
                        help="device to run model, `cpu` or `cuda:x`")
    parser.add_argument('--use_amp', dest='use_amp', default=False, action='store_true', 
                        help="whether to use amp")
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="batch size")
    parser.add_argument('--grad_clip', type=float, default=5.0, 
                        help="gradient clip (negative values are set to `None`)")
    
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD', 'Adadelta'], 
                        help="optimizer")
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="learning rate")
    parser.add_argument('--finetune_lr', type=float, default=2e-5, 
                        help="learning rate for finetuning")
    parser.add_argument('--scheduler', type=str, default='None', choices=['None', 'ReduceLROnPlateau', 'LinearDecayWithWarmup'], 
                        help='scheduler')
    
    parser.add_argument('--use_encoder', dest='use_encoder', default=False, action='store_true', help="whether to use ELMo")
    parser.add_argument('--emb_dim', type=int, default=100, help="embedding dim")
    parser.add_argument('--hid_dim', type=int, default=200, help="hidden dim")
    parser.add_argument('--num_layers', type=int, default=2, help="number of layers")
    parser.add_argument('--drop_rate', type=float, default=0.5, help="dropout rate")
    parser.add_argument('--emb_freeze', dest='emb_freeze', default=False, action='store_true', help="whether to freeze embedding weights")
    
    
    parser.add_argument('--dataset', type=str, default='conll2003', help="dataset name")
    parser.add_argument('--scheme', type=str, default='BIOES', help="sequence tagging scheme")
    parser.add_argument('--dec_arch', type=str, default='CRF', help="decoder architecture")
    parser.add_argument('--char_arch', type=str, default='CNN', help="character-level encoder")
    parser.add_argument('--use_elmo', dest='use_elmo', default=False, action='store_true', help="whether to use ELMo")
    parser.add_argument('--use_bert', dest='use_bert', default=False, action='store_true', help="whether to use BERT")
    
    parser.add_argument('--bert_drop_rate', type=float, default=0.2, help="dropout rate for BERT")
    parser.add_argument('--use_bert_intermediate', dest='use_bert_intermediate', default=False, action='store_true', help="whether to use BERT BiLSTM")
    parser.add_argument('--use_flair', dest='use_flair', default=False, action='store_true', help="whether to use Flair")
    args = parser.parse_args()
    args.grad_clip = None if args.grad_clip < 0 else args.grad_clip
    
    # Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path =  f"cache/{args.dataset}-{timestamp}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(level=logging.INFO, 
                        format="[%(asctime)s %(levelname)s] %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S", 
                        handlers=[logging.FileHandler(f"{save_path}/training.log"), 
                                  logging.StreamHandler(sys.stdout)])
    
    logger = logging.getLogger(__name__)
    logger.info(header_format("Starting"))
    logger.info(pprint.pformat(args.__dict__))
    
    # Preparing
    logger.info(header_format("Preparing", sep='='))
    device = torch.device(args.device)
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        
    glove = GloVe("assets/vectors/glove.6B.100d.txt")
    ohots_config = ConfigDict({'text': OneHotConfig(field='text', vectors=glove, emb_dim=100, freeze=args.emb_freeze)})
    encoder_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0))
    intermediate_config = None
    
    if args.char_arch.lower() in ('cnn', 'lstm', 'gru'):
        char_config = CharConfig(arch=args.char_arch, emb_dim=16, out_dim=128, pooling='Max', drop_rate=args.drop_rate)
    else:
        char_config = None
        
    elmo_config = None
    if args.use_elmo:
        elmo = allennlp.modules.Elmo(options_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_options.json", 
                                     weight_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", 
                                     num_output_representations=1)
        elmo_config = ELMoConfig(elmo=elmo)
        encoder_config = EncoderConfig(arch='Identity')
        intermediate_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0))
        
    bert_like_config = None
    if args.use_bert:
        # Cased tokenizer for NER task
        tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers/bert-base-cased")
        bert = transformers.BertModel.from_pretrained("assets/transformers/bert-base-cased", 
                                                      hidden_dropout_prob=args.bert_drop_rate, 
                                                      attention_probs_dropout_prob=0.2)
        bert_like_config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, use_truecase=True)
        ohots_config = None
        char_config = None
        encoder_config = None
        if args.use_bert_intermediate:
            intermediate_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0))
        
        
    flair_fw_config, flair_bw_config = None, None
    if args.use_flair:
        flair_fw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt")
        flair_bw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt")
        flair_fw_config = FlairConfig(flair_lm=flair_fw_lm)
        flair_bw_config = FlairConfig(flair_lm=flair_bw_lm)
        encoder_config = EncoderConfig(arch='Identity')
        intermediate_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(0.0, 0.05, args.drop_rate), 
                                            in_proj=True)
        
        
    config = SequenceTaggerConfig(ohots=ohots_config, 
                                  char=char_config,
                                  encoder=encoder_config, 
                                  elmo=elmo_config, 
                                  bert_like=bert_like_config, 
                                  flair_fw=flair_fw_config, 
                                  flair_bw=flair_bw_config, 
                                  intermediate=intermediate_config, 
                                  decoder=SequenceTaggingDecoderConfig(arch=args.dec_arch, scheme=args.scheme, in_drop_rates=(args.drop_rate, 0.0, 0.0)))
    train_data, dev_data, test_data = load_data(args)
    
    train_set = SequenceTaggingDataset(train_data, config)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    dev_set   = SequenceTaggingDataset(dev_data,  train_set.config)
    test_set  = SequenceTaggingDataset(test_data, train_set.config)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate)
    
    # Buiding the model
    logger.info(header_format("Building", sep='-'))
    tagger = config.instantiate().to(device)
    count_params(tagger)
    
    # Training
    logger.info(header_format("Training", sep='-'))
    def save_callback(model):
        torch.save(model, f"{save_path}/{args.scheme}-{config.name}.pth")
    
    param_groups = [{'params': tagger.pretrained_parameters(), 'lr': args.finetune_lr}]
    param_groups.append({'params': collect_params(tagger, param_groups), 'lr': args.lr})
    assert check_param_groups(tagger, param_groups)
    optimizer = getattr(torch.optim, args.optimizer)(param_groups)
    
    schedule_by_step = False
    if args.scheduler == 'None':
        scheduler = None
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        schedule_by_step = True
        
        # lr_lambda = LRLambda.constant_lr()
        num_warmup_epochs = max(2, args.num_epochs // 5)
        lr_lambda = LRLambda.linear_decay_lr_with_warmup(num_warmup_steps=len(train_loader)*num_warmup_epochs, 
                                                         num_total_steps=len(train_loader)*args.num_epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        
    trainer = SequenceTaggingTrainer(tagger, 
                                     optimizer=optimizer, scheduler=scheduler, schedule_by_step=schedule_by_step,
                                     device=device, grad_clip=args.grad_clip, use_amp=args.use_amp)
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    # Evaluating
    logger.info(header_format("Evaluating", sep='-'))
    tagger = torch.load(f"{save_path}/{args.scheme}-{config.name}.pth", map_location=device)
    trainer = SequenceTaggingTrainer(tagger, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_sequence_tagging(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_sequence_tagging(trainer, test_set)
    
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
    
    