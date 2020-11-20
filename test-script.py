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


from eznlp import Token, TokenSequence, count_params
from eznlp.token import Full2Half
from eznlp.dataset_utils import Batch
from eznlp import ConfigList, ConfigDict
from eznlp import CharConfig, TokenConfig, EnumConfig, ValConfig, EmbedderConfig
from eznlp import EncoderConfig
from eznlp import PreTrainedEmbedderConfig
from eznlp.vectors import Senna
from eznlp.nn import SequencePooling, SequenceGroupAggregating
from eznlp.sequence_tagging import DecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.sequence_tagging import ChunksTagsTranslator, SchemeTranslator
from eznlp.sequence_tagging import precision_recall_f1_report
from eznlp.sequence_tagging.raw_data import parse_conll_file
from eznlp.sequence_tagging.transition import find_ascending
from eznlp.language_modeling import MLMDataset, PMCMLMDataset, MLMTrainer

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
    batch_tokenized_text = [["I", "like", "it", "."], 
                            ["Do", "you", "love", "me", "?"], 
                            ["Sure", "!"], 
                            ["Future", "it", "out"]]
    
    batch_tok_lens = [[len(tok) for tok in sent] for sent in batch_tokenized_text]
    batch_text = [" ".join(sent) for sent in batch_tokenized_text]
    
    
    
    flair_fw_lm = LanguageModel.load_language_model("assets/flair/news-forward-0.4.1.pt")
    flair_bw_lm = LanguageModel.load_language_model("assets/flair/news-backward-0.4.1.pt")
    
    # (step, batch, hid_dim)
    # exp_flair_hidden = flair_lm.get_representation(batch_text, start_marker="\n", end_marker=" ")
    
    
    # options_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    # weight_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    # elmo = Elmo(options_file, weight_file, 1)
    # elmo._elmo_lstm._elmo_lstm.stateful = False
    # elmo.eval()
    
    # elmo_char_ids = batch_to_ids(batch_sentences)
    # elmo(elmo_char_ids)
    
    
    # bert = BertModel.from_pretrained("assets/transformers_cache/bert-base-cased")
    # tokenizer = BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-cased")
    
    # encoded = tokenizer(batch_sentences, is_pretokenized=True, padding=True, return_tensors='pt')
    # bert_outs, _, hidden = bert(**encoded, output_hidden_states=True)
    
    
    # glove = GloVe(name='6B', dim=100, root="assets/vector_cache", validate_file=False)
    
    conll_config = {'raw_scheme': 'BIO1', 
                    'scheme': 'BIOES', 
                    'columns': ['text', 'pos_tag', 'chunking_tag', 'ner_tag'], 
                    'trg_col': 'ner_tag', 
                    'attach_additional_tags': False, 
                    'skip_docstart': False, 
                    'lower_case_mode': 'None'}
    
    train_data = parse_conll_file("assets/data/conll2003/eng.train", max_examples=200, **conll_config)
    val_data   = parse_conll_file("assets/data/conll2003/eng.testa", max_examples=10,  **conll_config)
    test_data  = parse_conll_file("assets/data/conll2003/eng.testb", max_examples=10,  **conll_config)
    
    
    config = SequenceTaggerConfig(embedder=EmbedderConfig(
                                      token=TokenConfig(emb_dim=100), 
                                      char=CharConfig(arch='CNN', emb_dim=25, out_dim=50, drop_rate=0.5), 
                                      enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token.basic_enum_fields]), 
                                      val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token.basic_val_fields])
                                  ), 
                                  encoder=EncoderConfig(arch='LSTM', hid_dim=200, num_layers=1, shortcut=True),
                                  # elmo_embedder=PreTrainedEmbedderConfig(arch='ELMo', out_dim=elmo.get_output_dim(), freeze=True), 
                                  # bert_like_embedder=PreTrainedEmbedderConfig(arch='BERT', out_dim=bert.config.hidden_size, tokenizer=tokenizer, freeze=True), 
                                  flair_fw_embedder=PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_fw_lm.hidden_size, freeze=True), 
                                  flair_bw_embedder=PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_bw_lm.hidden_size, freeze=True),
                                  intermediate=EncoderConfig(), 
                                  decoder =DecoderConfig(arch='CRF'))
    train_set = SequenceTaggingDataset(train_data, config)
    val_set   = SequenceTaggingDataset(val_data,   train_set.config)
    test_set  = SequenceTaggingDataset(test_data,  train_set.config)
    
    tagger = config.instantiate(flair_fw_lm=flair_fw_lm, flair_bw_lm=flair_bw_lm)
    
    batch = train_set.collate([train_set[i] for i in range(0, 4)])
    losses, hidden = tagger(batch, return_hidden=True)
    
    
    