# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import random
import logging
import numpy as np
import torch

from eznlp.config import ConfigDict
from eznlp.model import CharConfig, OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.pretrained import Vectors
from eznlp.training.utils import count_params

from script_utils import parse_basic_arguments, load_data, evaluate_sequence_tagging


SEED = 515
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = parse_basic_arguments(parser)
    
    parser.add_argument('--dataset', type=str, default='ResumeNER', help="dataset name")
    parser.add_argument('--scheme', type=str, default='BIOES', help="sequence tagging scheme")
    parser.add_argument('--use_bigram', dest='use_bigram', default=False, action='store_true', help="whether to use bigram")
    parser.add_argument('--use_softword', dest='use_softword', default=False, action='store_true', help="whether to use softword")
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
    logger.info("========== Starting ==========")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Use bigram: {args.use_bigram}")
    logger.info(f"Use softword: {args.use_softword}")
    
    
    # Preparing
    logger.info("---------- Preparing ----------")
    device = torch.device(args.device)
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        
        
    giga_uni = Vectors.load("assets/vectors/gigaword_chn.all.a2b.uni.ite50.vec", encoding='utf-8')
    ohots_config = ConfigDict({'text': OneHotConfig(field='text', vectors=giga_uni, emb_dim=50, freeze=args.emb_freeze)})
    if args.use_bigram:
        giga_bi = Vectors.load("assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec", encoding='utf-8')
        ohots_config['bigram'] = OneHotConfig(field='bigram', vectors=giga_bi, emb_dim=50, freeze=args.emb_freeze)
    
    mhots_config = None
    if args.use_softword:
        mhots_config = ConfigDict({'softword': MultiHotConfig(field='softword', emb_dim=20)})
        
    config = SequenceTaggerConfig(ohots=ohots_config, 
                                  mhots=mhots_config,
                                  encoder=EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0)), 
                                  decoder=SequenceTaggingDecoderConfig(arch='CRF', scheme=args.scheme, in_drop_rates=(args.drop_rate, 0.0, 0.0)))
    train_data, dev_data, test_data = load_data(args)
    
    train_set = SequenceTaggingDataset(train_data, config)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    dev_set   = SequenceTaggingDataset(dev_data,  train_set.config)
    test_set  = SequenceTaggingDataset(test_data, train_set.config)
    
    logger.info(train_set.summary)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, collate_fn=train_set.collate)
    dev_loader   = torch.utils.data.DataLoader(dev_set,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=dev_set.collate)
    
    # Buiding the model
    logger.info("---------- Building the model ----------")
    tagger = config.instantiate().to(device)
    count_params(tagger)
    
    if args.debug:
        import pdb; pdb.set_trace()
    
    # Training
    logger.info("---------- Training ----------")
    def save_callback(model):
        torch.save(model, f"{save_path}/{args.scheme}-{config.name}.pth")
        
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(tagger.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(tagger.parameters(), lr=args.lr)
        
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    # scheduler = None
    
    trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, scheduler=scheduler, device=device, grad_clip=args.grad_clip)
    trainer.train_steps(train_loader=train_loader, dev_loader=dev_loader, num_epochs=args.num_epochs, 
                        save_callback=save_callback, save_by_loss=False)
    
    # Evaluating
    logger.info("---------- Evaluating ----------")
    tagger = torch.load(f"{save_path}/{args.scheme}-{config.name}.pth", map_location=device)
    trainer = SequenceTaggingTrainer(tagger, device=device)
    
    logger.info("Evaluating on dev-set")
    evaluate_sequence_tagging(trainer, dev_set)
    logger.info("Evaluating on test-set")
    evaluate_sequence_tagging(trainer, test_set)
    
    logger.info("========== Ending ==========")
    
    