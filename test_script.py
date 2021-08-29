# -*- coding: utf-8 -*-
from collections import OrderedDict, Counter
import argparse
import logging
import glob
import pickle
import numpy as np
import pandas as pd
import spacy
import jieba
import torch
import torchtext
import torchvision

import allennlp.modules
import transformers
import flair

from eznlp import auto_device
from eznlp.utils import find_ascending, ChunksTagsTranslator
from eznlp.token import Token, TokenSequence, LexiconTokenizer
from eznlp.metrics import precision_recall_f1_report
from eznlp.vectors import Vectors, GloVe, Senna
from eznlp.wrapper import Batch

from eznlp.io import TabularIO, CategoryFolderIO, ConllIO, BratIO, JsonIO, SQuADIO, ChipIO
from eznlp.io import PostIO

from eznlp.dataset import Dataset

from eznlp.config import ConfigList, ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.model import NestedOneHotConfig, CharConfig, SoftLexiconConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model import ImageEncoderConfig
from eznlp.model.bert_like import truncate_for_bert_like

from eznlp.model import (TextClassificationDecoderConfig, 
                         SequenceTaggingDecoderConfig, 
                         SpanClassificationDecoderConfig, 
                         SpanAttrClassificationDecoderConfig, 
                         SpanRelClassificationDecoderConfig, 
                         BoundarySelectionDecoderConfig, 
                         JointExtractionDecoderConfig, 
                         GeneratorConfig)
from eznlp.model import ExtractorConfig, Image2TextConfig

from eznlp.language_modeling import MaskedLMConfig
from eznlp.language_modeling import MaskedLMDataset, FolderLikeMaskedLMDataset, MaskedLMTrainer

from eznlp.training import Trainer
from eznlp.training import collect_params, check_param_groups



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = auto_device()
    
    # with open("data/multi30k/train.de", encoding='utf-8') as f:
    #     txt = f.readlines()
    #     print(len(txt))

    # batch_tokenized_raw_text = [["I", "like", "it", "."], 
    #                             ["Do", "you", "love", "me", "?"], 
    #                             ["Sure", "!"], 
    #                             ["Future", "it", "out"]]
    
    
    # flair_fw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt")
    # flair_bw_lm = flair.models.LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt")
    
    # options_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    # weight_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    # elmo = allennlp.modules.Elmo(options_file, weight_file, 1)
    # batch_elmo_ids = allennlp.modules.elmo.batch_to_ids(batch_tokenized_raw_text)
    
    # bert = transformers.BertModel.from_pretrained("assets/transformers/bert-base-uncased")
    # tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers/bert-base-uncased")
    
    # glove = GloVe("assets/vectors/glove.6B.100d.txt", encoding='utf-8')
    # senna = Senna("assets/vectors/Senna")
    
    # ctb50d = Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
    # giga_uni = Vectors.load("assets/vectors/gigaword_chn.all.a2b.uni.ite50.vec", encoding='utf-8')
    # giga_bi  = Vectors.load("assets/vectors/gigaword_chn.all.a2b.bi.ite50.vec", encoding='utf-8')
    # # tencent = Vectors.load("assets/vectors/tencent/Tencent_AILab_ChineseEmbedding.txt", encoding='utf-8', skiprows=0, verbose=True)
    
    # conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO2')
    # train_data = conll_io.read("data/conll2003/demo.eng.train")
    # dev_data   = conll_io.read("data/conll2003/demo.eng.testa")
    # test_data  = conll_io.read("data/conll2003/demo.eng.testb")
    
    # config = ExtractorConfig('sequence_tagging', 
    #                          ohots=ConfigDict({'text': OneHotConfig(field='text', vectors=glove)}), 
    #                          nested_ohots=ConfigDict({'char': CharConfig()}), 
    #                          elmo=ELMoConfig(elmo=elmo), 
    #                          bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
    #                          flair_fw=FlairConfig(flair_lm=flair_fw_lm), 
    #                          flair_bw=FlairConfig(flair_lm=flair_bw_lm))
    
    # train_set = Dataset(train_data, config)
    # train_set.build_vocabs_and_dims(dev_data, test_data)
    # model = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, states = model(batch, return_states=True)
    
    # optimizer = torch.optim.AdamW(model.parameters())
    # trainer = Trainer(model, optimizer=optimizer, device=device)
    # res = trainer.train_epoch([batch])
    
    caption_io = TabularIO(text_col_id=1, label_col_id=0, sep='\t')
    data = caption_io.read("data/flickr8k/Flickr8k.token.txt")
    for entry in data:
        entry['trg_tokens'] = entry.pop('tokens')
        entry['img_fn'], entry['cap_no'] = entry.pop('label').split('#')
    
    img_folder = "data/flickr8k/Flicker8k_Dataset"
    
    with open("data/flickr8k/Flickr_8k.trainImages.txt") as f:
        train_fns = set([line.strip() for line in f])
    with open("data/flickr8k/Flickr_8k.devImages.txt") as f:
        dev_fns = set([line.strip() for line in f])
    with open("data/flickr8k/Flickr_8k.testImages.txt") as f:
        test_fns = set([line.strip() for line in f])
        
    train_data = [entry for entry in data if entry['img_fn'] in train_fns]
    dev_data   = [entry for entry in data if entry['img_fn'] in dev_fns]
    test_data  = [entry for entry in data if entry['img_fn'] in test_fns]
    
    
    # i = 0
    # entry = train_data[i]
    # img = torchvision.io.read_image(f"{img_folder}/{entry['img_fn']}")
    # # torchvision.transforms.functional.to_pil_image(img)
    # img = img / 255.0
    
    resnet = torchvision.models.resnet101(pretrained=False)
    resnet.load_state_dict(torch.load("assets/resnet/resnet101-5d3b4d8f.pth"))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    
    trans = torch.nn.Sequential(torchvision.transforms.Resize((256, 256)), 
                                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std =[0.229, 0.224, 0.225]))
    
    config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, folder=img_folder, transforms=trans), 
                              decoder=GeneratorConfig())
    train_set = Dataset(train_data, config, training=True)
    train_set.build_vocabs_and_dims(dev_data, test_data)
    
    dev_data  = pd.DataFrame(dev_data,  columns=['img_fn', 'trg_tokens']).groupby('img_fn').aggregate(lambda x: x.tolist()).reset_index().to_dict(orient='records')
    test_data = pd.DataFrame(test_data, columns=['img_fn', 'trg_tokens']).groupby('img_fn').aggregate(lambda x: x.tolist()).reset_index().to_dict(orient='records')
    dev_set   = Dataset(dev_data,  config=train_set.config, training=False)
    test_set  = Dataset(test_data, config=train_set.config, training=False)
    model = config.instantiate()
    
    batch = train_set.collate([train_set[i] for i in range(0, 4)])
    losses, states = model(batch, return_states=True)
    
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, optimizer=optimizer, device=device)
    res = trainer.train_epoch([batch])
    print(res)
    
    
    # for model_name in ["hfl/chinese-macbert-base", "hfl/chinese-macbert-large"]:
    #     logging.info(f"Start downloading {model_name}...")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    #     model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    #     tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    #     model.save_pretrained(f"assets/transformers/{model_name}")
    