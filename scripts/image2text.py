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
import torchvision

from eznlp import auto_device
from eznlp.dataset import GenerationDataset
from eznlp.model import ImageEncoderConfig, OneHotConfig, GeneratorConfig
from eznlp.model import Image2TextConfig
from eznlp.training import Trainer, count_params, evaluate_generation

from utils import add_base_arguments, parse_to_args
from utils import load_data, dataset2language, load_vectors, build_trainer, header_format



def parse_arguments(parser: argparse.ArgumentParser):
    parser = add_base_arguments(parser)
    
    group_data = parser.add_argument_group('dataset')
    group_data.add_argument('--dataset', type=str, default='flickr8k', 
                            help="dataset name")
    
    group_image = parser.add_argument_group('image encoder configurations')
    group_image.add_argument('--img_arch', type=str, default='ResNet', choices=['VGG', 'ResNet'], 
                             help="image encoder architecture")
    group_image.add_argument('--use_cache', default=False, action='store_true', 
                             help="whether to use cache for images")
    
    group_decoder = parser.add_argument_group('decoder configurations')
    group_decoder.add_argument('--dec_arch', type=str, default='LSTM', choices=['LSTM', 'GRU', 'Gehring', 'Transformer'], 
                               help="token-level decoder architecture")
    group_decoder.add_argument('--atten_num_heads', type=int, default=1, 
                               help="attention number of heads")
    group_decoder.add_argument('--atten_scoring', type=str, default='biaffine', 
                               help="attention scoring")
    group_decoder.add_argument('--sin_positional_emb', default=False, action='store_true', 
                               help="whether to use sinusoid positional encodings")
    group_decoder.add_argument('--use_weight_tying', default=False, action='store_true', 
                               help="whether to use weight tying")
    group_decoder.add_argument('--teacher_forcing_rate', type=float, default=0.5, 
                               help="teacher forcing rate")
    group_decoder.add_argument('--init_ctx_mode', type=str, default='mean_pooling', 
                               help="rnn init context vector mode")
    group_decoder.add_argument('--ff_dim', type=int, default=512, 
                               help="transformer position-wise feedforward dim")
    # Loss
    group_decoder.add_argument('--fl_gamma', type=float, default=0.0, 
                               help="focal Loss gamma")
    group_decoder.add_argument('--sl_epsilon', type=float, default=0.0, 
                               help="label smoothing loss epsilon")
    return parse_to_args(parser)



def collect_I2T_assembly_config(args: argparse.Namespace):
    drop_rates = (0.0, 0.05, args.drop_rate) if args.use_locked_drop else (args.drop_rate, 0.0, 0.0)
    
    if args.img_arch.lower() == 'vgg':
        backbone = torchvision.models.vgg19(pretrained=False)
        backbone.load_state_dict(torch.load("assets/vgg/vgg19-dcbb9e9d.pth"))
    else:
        backbone = torchvision.models.resnet101(pretrained=False)
        backbone.load_state_dict(torch.load("assets/resnet/resnet101-5d3b4d8f.pth"))
    
    # https://pytorch.org/vision/stable/models.html
    trans = torch.nn.Sequential(torchvision.transforms.Resize(256), 
                                torchvision.transforms.CenterCrop(224), 
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std =[0.229, 0.224, 0.225]))
    enc_config = ImageEncoderConfig(arch=args.img_arch, backbone=backbone, transforms=trans, use_cache=args.use_cache)
    
    vectors = load_vectors(args.language, args.emb_dim)
    emb_config = OneHotConfig(tokens_key='trg_tokens', field='text', min_freq=2, has_sos=True, has_eos=True, 
                              vectors=vectors, emb_dim=args.emb_dim, freeze=args.emb_freeze, 
                              has_positional_emb=(args.dec_arch.lower() not in ('lstm', 'gru')), sin_positional_emb=args.sin_positional_emb)
    gen_config = GeneratorConfig(arch=args.dec_arch, embedding=emb_config, hid_dim=args.hid_dim, num_layers=args.num_layers, in_drop_rates=drop_rates, 
                                 fl_gamma=args.fl_gamma, sl_epsilon=args.sl_epsilon, 
                                 num_heads=args.atten_num_heads, scoring=args.atten_scoring, 
                                 weight_tying=args.use_weight_tying, teacher_forcing_rate=args.teacher_forcing_rate, 
                                 init_ctx_mode=args.init_ctx_mode, ff_dim=args.ff_dim)
    
    return {'encoder': enc_config, 
            'decoder': gen_config}


def build_I2T_config(args: argparse.Namespace):
    return Image2TextConfig(**collect_I2T_assembly_config(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    args = parse_arguments(parser)
    
    # Use micro-seconds to ensure different timestamps while adopting multiprocessing
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    save_path =  f"cache/{args.dataset}-I2T/{timestamp}"
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
    # train_data, dev_data, test_data = train_data[:100], dev_data[:100], test_data[:100]
    config = build_I2T_config(args)
    
    train_set = GenerationDataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims()
    # train_set.build_vocabs_and_dims(dev_data)
    dev_set   = GenerationDataset(dev_data,  config=train_set.config, training=False)
    test_set  = GenerationDataset(test_data, config=train_set.config, training=False)
    
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
    evaluate_generation(trainer, dev_set, batch_size=args.batch_size, beam_size=1)
    evaluate_generation(trainer, dev_set, batch_size=args.batch_size, beam_size=2)
    evaluate_generation(trainer, dev_set, batch_size=args.batch_size, beam_size=3)
    evaluate_generation(trainer, dev_set, batch_size=args.batch_size, beam_size=4)
    logger.info("Evaluating on test-set")
    evaluate_generation(trainer, test_set, batch_size=args.batch_size, beam_size=1)
    evaluate_generation(trainer, test_set, batch_size=args.batch_size, beam_size=2)
    evaluate_generation(trainer, test_set, batch_size=args.batch_size, beam_size=3)
    evaluate_generation(trainer, test_set, batch_size=args.batch_size, beam_size=4)
    
    logger.info(" ".join(sys.argv))
    logger.info(pprint.pformat(args.__dict__))
    logger.info(header_format("Ending", sep='='))
