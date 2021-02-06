# -*- coding: utf-8 -*-
import os
import sys
import argparse
import datetime
import random
import logging
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
from eznlp.training.utils import count_params

from script_utils import load_data, evaluate_sequence_tagging


SEED = 515
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', default='cpu', help="Device to run the model, `cpu` or `cuda:x`")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--drop_rate', type=float, default=0.5, help="Dropout rate")
    parser.add_argument('--grad_clip', type=float, default=5.0, help="Gradient clip")
    
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--hid_dim', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=2)
    
    parser.add_argument('--dataset', default='conll2003', help="Dataset name")
    parser.add_argument('--scheme', default='BIOES', help="Sequence tagging scheme")
    parser.add_argument('--use_char', type=bool, default=True)
    parser.add_argument('--use_elmo', type=bool, default=False)
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--use_flair', type=bool, default=False)
    args = parser.parse_args()
    
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
    logger.info(f"Use char: {args.use_char}")
    
    
    # Preparing
    logger.info("---------- Preparing ----------")
    device = torch.device(args.device)
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device)
        
    glove = GloVe("assets/vectors/glove.6B.100d.txt")
    ohots_config = ConfigDict({'text': OneHotConfig(field='text', vectors=glove, emb_dim=100)})
    encoder_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0))
    intermediate_config = None
    
    char_config = None
    if args.use_char:
        char_config = CharConfig(arch='CNN', emb_dim=16, out_dim=128, pooling='Max', drop_rate=0.5)
    
    elmo_config = None
    if args.use_elmo:
        elmo = allennlp.modules.Elmo(options_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_options.json", 
                                     weight_file="assets/allennlp/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", 
                                     num_output_representations=1)
        elmo_config = ELMoConfig(elmo=elmo)
        encoder_config = EncoderConfig(arch='Identity')
        intermediate_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(args.drop_rate, 0.0, 0.0))
        
    flair_fw_config, flair_bw_config = None, None
    if args.use_flair:
        flair_fw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt")
        flair_bw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt")
        flair_fw_config = FlairConfig(flair_lm=flair_fw_lm)
        flair_bw_config = FlairConfig(flair_lm=flair_bw_lm)
        char_config = None
        encoder_config = EncoderConfig(arch='Identity')
        intermediate_config = EncoderConfig(arch='LSTM', hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=(0.0, 0.05, args.drop_rate), 
                                            in_proj=True)
        
        
    config = SequenceTaggerConfig(ohots=ohots_config, 
                                  char=char_config,
                                  encoder=encoder_config, 
                                  elmo=elmo_config, 
                                  flair_fw=flair_fw_config, 
                                  flair_bw=flair_bw_config, 
                                  intermediate=intermediate_config, 
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
    
    # import pdb; pdb.set_trace()
    
    # Training
    logger.info("---------- Training ----------")
    def save_callback(model):
        torch.save(model, f"{save_path}/{args.scheme}-{config.name}.pth")
        
        
    # optimizer = torch.optim.SGD(tagger.parameters(), lr=args.lr)
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
    
    