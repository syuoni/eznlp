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

import allennlp.modules
import transformers
import flair

from eznlp import auto_device
from eznlp.utils import find_ascending
from eznlp.token import Token, TokenSequence, LexiconTokenizer
from eznlp.metrics import precision_recall_f1_report
from eznlp.vectors import Vectors, GloVe, Senna
from eznlp.wrapper import Batch

from eznlp.io import TabularIO, CategoryFolderIO, ConllIO, BratIO, JsonIO

from eznlp.dataset import Dataset

from eznlp.config import ConfigList, ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.model import NestedOneHotConfig, CharConfig, SoftLexiconConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model.bert_like import truncate_for_bert_like
from eznlp.model.decoder.transition import ChunksTagsTranslator

from eznlp.model import (TextClassificationDecoderConfig, 
                         SequenceTaggingDecoderConfig, 
                         SpanClassificationDecoderConfig, 
                         RelationClassificationDecoderConfig, 
                         JointDecoderConfig)
from eznlp.model import ModelConfig

from eznlp.language_modeling import MaskedLMConfig
from eznlp.language_modeling import MaskedLMDataset, FolderLikeMaskedLMDataset, MaskedLMTrainer

from eznlp.training import Trainer
from eznlp.training.utils import collect_params, check_param_groups



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = auto_device()
    
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
    
    # config = ModelConfig('sequence_tagging', 
    #                       ohots=ConfigDict({'text': OneHotConfig(field='text', vectors=glove)}), 
    #                       nested_ohots=ConfigDict({'char': CharConfig()}), 
    #                       elmo=ELMoConfig(elmo=elmo), 
    #                       bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
    #                       flair_fw=FlairConfig(flair_lm=flair_fw_lm), 
    #                       flair_bw=FlairConfig(flair_lm=flair_bw_lm))
    
    # train_set = Dataset(train_data, config)
    # train_set.build_vocabs_and_dims(dev_data, test_data)
    # model = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, hidden = model(batch, return_hidden=True)
    
    # optimizer = torch.optim.AdamW(model.parameters())
    # trainer = Trainer(model, optimizer=optimizer, device=device)
    # res = trainer.train_epoch([batch])
    
    
    
    # brat_io = BratIO(attr_names=['Denied', 'Analyzed'], tokenize_callback=jieba.cut, max_len=50, encoding='utf-8')
    # brat_data = brat_io.read("data/brat/demo.txt")
    # brat_io.write(brat_data, "data/brat/demo-write.txt")
    
    # brat_set = Dataset(brat_data, ModelConfig('sequence_tagging'))
    # brat_set.build_vocabs_and_dims()
    # batch = brat_set.collate([brat_set[i] for i in range(0, 4)])
    
    # model = brat_set.config.instantiate()
    # losses, hidden = model(batch, return_hidden=True)
    
    
    
    # tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8')
    # train_data = tabular_io.read("data/Tang2015/demo.yelp-2013-seg-20-20.train.ss")
    
    # config = ModelConfig('text_classification', 
    #                       ohots=None, 
    #                       intermediate2=None,
    #                       bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert))
    
    # train_set = Dataset(train_data, config)
    # train_set.build_vocabs_and_dims()
    # model = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, hidden = model(batch, return_hidden=True)
    
    # optimizer = torch.optim.AdamW(model.parameters())
    # trainer = Trainer(model, optimizer=optimizer, device=device)
    # trainer.train_epoch([batch])
    
    # json_io = JsonIO(text_key='tokens', 
    #                  chunk_key='entities', 
    #                  chunk_type_key='type', 
    #                  chunk_start_key='start', 
    #                  chunk_end_key='end', 
    #                  relation_key='relations', 
    #                  relation_type_key='type', 
    #                  relation_head_key='head', 
    #                  relation_tail_key='tail')
    # train_data = json_io.read("data/conll2004/conll04_train.json")
    
    # config = ModelConfig('joint')
    # train_set = Dataset(train_data, config)
    # train_set.build_vocabs_and_dims()
    
    # model = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(4)])
    # model(batch)
    # model.decode(batch)
    
    # for model_name in ["hfl/chinese-bert-wwm-ext", 
    #                     "hfl/chinese-roberta-wwm-ext"]:
    #     logging.info(f"Start downloading {model_name}...")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    #     model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    #     tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    #     model.save_pretrained(f"assets/transformers/{model_name}")
    