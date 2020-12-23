# -*- coding: utf-8 -*-
import time
from collections import OrderedDict, Counter
import glob
import pickle
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.experimental.vectors import Vectors, GloVe
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM


from eznlp.data import Token, TokenSequence
from eznlp.data import Batch
from eznlp import ConfigList, ConfigDict
from eznlp import CharConfig, TokenConfig, EnumConfig, ValConfig, EmbedderConfig
from eznlp import EncoderConfig
from eznlp import PreTrainedEmbedderConfig
from eznlp.vectors import Senna
from eznlp.nn import SequencePooling, SequenceGroupAggregating
from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.sequence_tagging import ChunksTagsTranslator
from eznlp.sequence_tagging import precision_recall_f1_report
from eznlp.sequence_tagging.io import ConllIO, BratIO
from eznlp.sequence_tagging.transition import find_ascending

from eznlp.language_modeling import MLMDataset, PMCMLMDataset, MLMTrainer

from eznlp.text_classification.io import TabularIO
from eznlp.text_classification import TextClassifierConfig
from eznlp.text_classification import TextClassificationDataset
from eznlp.text_classification import TextClassificationTrainer

from seqeval.metrics import classification_report
from torchcrf import CRF
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from flair.data import Sentence, Corpus
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, FlairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings
from flair.models import LanguageModel, SequenceTagger
from flair.trainers import ModelTrainer


if __name__ == '__main__':
    device = torch.device('cpu')
    
    # batch_tokenized_text = [["I", "like", "it", "."], 
    #                         ["Do", "you", "love", "me", "?"], 
    #                         ["Sure", "!"], 
    #                         ["Future", "it", "out"]]
    
    # batch_tok_lens = [[len(tok) for tok in sent] for sent in batch_tokenized_text]
    # batch_text = [" ".join(sent) for sent in batch_tokenized_text]
    
    # flair_fw_lm = LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt")
    # flair_bw_lm = LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt")
    
    # (step, batch, hid_dim)
    # exp_flair_hidden = flair_lm.get_representation(batch_text, start_marker="\n", end_marker=" ")
    
    
    # options_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    # weight_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    # elmo = Elmo(options_file, weight_file, 1)
    # elmo._elmo_lstm._elmo_lstm.stateful = False
    # elmo.eval()
    
    # elmo_char_ids = batch_to_ids(batch_sentences)
    # elmo(elmo_char_ids)
    
    
    bert = BertModel.from_pretrained("assets/transformers_cache/bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-uncased")
    
    # encoded = tokenizer(batch_sentences, is_pretokenized=True, padding=True, return_tensors='pt')
    # bert_outs, _, hidden = bert(**encoded, output_hidden_states=True)
    
    
    # glove = GloVe(name='6B', dim=100, root="assets/vector_cache", validate_file=False)
    
    # conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', additional_col_id2name={1: 'pos_tag'})
    # train_data = conll_io.read("assets/data/conll2003/eng.train")
    # val_data   = conll_io.read("assets/data/conll2003/eng.testa")
    # test_data  = conll_io.read("assets/data/conll2003/eng.testb")
    
    # config = SequenceTaggerConfig(embedder=EmbedderConfig(
    #                                   token=TokenConfig(emb_dim=100), 
    #                                   char=CharConfig(arch='CNN', emb_dim=25, out_dim=50, drop_rate=0.5), 
    #                                   enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token.basic_enum_fields]), 
    #                                   val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token.basic_val_fields])
    #                               ), 
    #                               encoder=EncoderConfig(arch='LSTM', hid_dim=200, num_layers=1, shortcut=True),
    #                               # elmo_embedder=PreTrainedEmbedderConfig(arch='ELMo', out_dim=elmo.get_output_dim(), freeze=True), 
    #                               # bert_like_embedder=PreTrainedEmbedderConfig(arch='BERT', out_dim=bert.config.hidden_size, tokenizer=tokenizer, freeze=True), 
    #                               # flair_fw_embedder=PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_fw_lm.hidden_size, freeze=True), 
    #                               # flair_bw_embedder=PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_bw_lm.hidden_size, freeze=True),
    #                               intermediate=EncoderConfig(), 
    #                               decoder=SequenceTaggingDecoderConfig(arch='CRF'))
    # train_set = SequenceTaggingDataset(train_data, config)
    # val_set   = SequenceTaggingDataset(val_data,   train_set.config)
    # test_set  = SequenceTaggingDataset(test_data,  train_set.config)
    
    # tagger = config.instantiate(flair_fw_lm=flair_fw_lm, flair_bw_lm=flair_bw_lm)
    
    # batch = train_set.collate([train_set[i] for i in range(0, 4)])
    # losses, hidden = tagger(batch, return_hidden=True)
    
    
    # brat_io = BratIO(use_attrs=['Denied', 'Analyzed'])
    # brat_data = brat_io.read("assets/data/brat/demo.txt", encoding='utf-8')
    # brat_io.write(brat_data, "assets/data/brat/demo-write.txt", encoding='utf-8')
    
    tabular_io = TabularIO(text_col_id=3, label_col_id=2)
    train_data = tabular_io.read("assets/data/Tang2015/yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
    
    config = TextClassifierConfig(encoder=None, 
                                  bert_like_embedder=PreTrainedEmbedderConfig(arch='BERT', out_dim=bert.config.hidden_size, 
                                                                              tokenizer=tokenizer, freeze=True))
    train_set = TextClassificationDataset(train_data, config)
    
    classifier = config.instantiate(bert_like=bert)
    
    batch = train_set.collate([train_set[i] for i in range(0, 4)])
    losses, hidden = classifier(batch, return_hidden=True)
    
    # optimizer = optim.AdamW(classifier.parameters())
    # trainer = TextClassificationTrainer(classifier, optimizer=optimizer, device=device)
    # trainer.train_epoch([batch])
    