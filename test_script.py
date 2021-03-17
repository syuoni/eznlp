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

import allennlp.modules
import transformers
import flair

from eznlp import auto_device
from eznlp.config import ConfigList, ConfigDict
from eznlp.token import Token, TokenSequence, LexiconTokenizer
from eznlp.data import Batch
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.model import NestedOneHotConfig, CharConfig, SoftLexiconConfig
from eznlp.pretrained import Vectors, GloVe, Senna
from eznlp.pretrained import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.training.utils import collect_params, check_param_groups


from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.sequence_tagging import ChunksTagsTranslator
from eznlp.sequence_tagging import precision_recall_f1_report
from eznlp.sequence_tagging.io import ConllIO, BratIO
from eznlp.sequence_tagging.transition import find_ascending

from eznlp.language_modeling import MaskedLMConfig
from eznlp.language_modeling import MaskedLMDataset, FolderLikeMaskedLMDataset, MaskedLMTrainer

from eznlp.text_classification.io import TabularIO
from eznlp.text_classification import TextClassificationDecoderConfig, TextClassifierConfig
from eznlp.text_classification import TextClassificationDataset
from eznlp.text_classification import TextClassificationTrainer



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = auto_device()
    
    ctb50 = Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')
    tokenizer = LexiconTokenizer(ctb50.itos)
    
    # import jieba
    data = [{'tokens': TokenSequence.from_tokenized_text(list("李明住在中山西路。"), token_sep="")}, 
            {'tokens': TokenSequence.from_tokenized_text(list("我爱北京天安门！"), token_sep="")}]
    for data_entry in data:
        data_entry['tokens'].build_softlexicons(tokenizer.tokenize)
    
    
    config = NestedOneHotConfig(field='softlexicon', num_channels=4, squeeze=False)
    config.build_vocab(data)
    
    
    
    # config.batchify([config.exemplify(data[0]['tokens']), 
    #                  config.exemplify(data[1]['tokens'])])
    
    # embedder = config.instantiate()
    # embedder
    
    
    # print(list(jieba.cut_for_search("".join(tokens.raw_text))))
    
    # for tok in jieba.tokenize("".join(tokens.raw_text), mode='search'):
    #     print(tok)
        
    # tokens.build_softwords(jieba.tokenize)
    
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
    # giga_uni = load_vectors_from_file("assets/vector_cache/gigaword_chn.all.a2b.uni.ite50.vec", encoding='utf-8')
    # giga_bi  = load_vectors_from_file("assets/vector_cache/gigaword_chn.all.a2b.bi.ite50.vec", encoding='utf-8')
    
    
    conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO2')
    train_data = conll_io.read("data/conll2003/demo.eng.train")
    dev_data   = conll_io.read("data/conll2003/demo.eng.testa")
    test_data  = conll_io.read("data/conll2003/demo.eng.testb")
    
    
    config = NestedOneHotConfig(field='raw_text', encoder=EncoderConfig())
    config.build_vocab(train_data)
    
    
    
    
    
    # config = SequenceTaggerConfig(ohots=ConfigDict({'text': OneHotConfig(field='text', vectors=glove)}), 
    #                               char=CharConfig(),
    #                               elmo=ELMoConfig(elmo=elmo), 
    #                               bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
    #                               flair_fw=FlairConfig(flair_lm=flair_fw_lm), 
    #                               flair_bw=FlairConfig(flair_lm=flair_bw_lm))
    
    # train_set = SequenceTaggingDataset(train_data, config)
    # train_set.build_vocabs_and_dims(dev_data, test_data)
    # tagger = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, hidden = tagger(batch, return_hidden=True)
    
    # param_groups = [{'params': list(tagger.decoder.crf.parameters())}, 
    #                 {'params': list(tagger.ohots['text'].parameters())}]
    # param_groups.append({'params': collect_params(tagger, param_groups)})
    # check_param_groups(tagger, param_groups)
    # optimizer = torch.optim.AdamW(param_groups)
    
    
    # brat_io = BratIO(attr_names=['Denied', 'Analyzed'], tokenize_callback=jieba.cut)
    # brat_data = brat_io.read("assets/data/brat/demo.txt", encoding='utf-8')
    # brat_io.write(brat_data, "assets/data/brat/demo-write.txt", encoding='utf-8')
    
    # brat_set = SequenceTaggingDataset(brat_data)
    # batch = brat_set.collate([brat_set[i] for i in range(0, 4)])
    
    # tagger = brat_set.config.instantiate()
    # losses = tagger(batch)
    # optimizer = torch.optim.AdamW(tagger.parameters())
    # trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, device=device)
    # res = trainer.train_epoch([batch])
    
    # param_groups = [{'params': list(tagger.decoder.crf.parameters())}]
    
    
    # tabular_io = TabularIO(text_col_id=3, label_col_id=2)
    # train_data = tabular_io.read("data/Tang2015/demo.yelp-2013-seg-20-20.train.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
    
    # config = TextClassifierConfig(ohots=None, 
    #                               encoder=None, 
    #                               bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert))
    
    # train_set = TextClassificationDataset(train_data, config)
    # train_set.build_vocabs_and_dims()
    
    # classifier = config.instantiate()
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, hidden = classifier(batch, return_hidden=True)
    
    # optimizer = torch.optim.AdamW(classifier.parameters())
    # trainer = TextClassificationTrainer(classifier, optimizer=optimizer, device=device)
    # trainer.train_epoch([batch])
    
    # for model_name in ["hfl/chinese-bert-wwm-ext", 
    #                    "hfl/chinese-roberta-wwm-ext"]:
    #     logging.info(f"Start downloading {model_name}...")
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    #     model = transformers.AutoModelForPreTraining.from_pretrained(model_name)
    #     tokenizer.save_pretrained(f"assets/transformers/{model_name}")
    #     model.save_pretrained(f"assets/transformers/{model_name}")
        