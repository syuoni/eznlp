# -*- coding: utf-8 -*-
import unittest
import glob
import pickle
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.experimental.vectors import GloVe
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM


from eznlp import Token, TokenSequence, build_token_sequence, count_trainable_params
from eznlp.token import Full2Half
from eznlp.ner import COVID19Dataset
from eznlp.ner import ConfigHelper
from eznlp.ner import build_tagger_by_config
from eznlp.ner import NERTrainer
from eznlp.ner import entities2tags, tags2entities
from eznlp.ner.datasets import TagHelper
from eznlp.ner.data_utils import find_ascending, tags2simple_entities
from eznlp.lm import COVID19MLMDataset, PMCMLMDataset, MLMTrainer


class TestFindAscending(unittest.TestCase):
    def test_find_ascending(self):
        for v in [-3, 2, 2.5, 9]:
            x = list(range(5))
            find, idx = find_ascending(x, v)
            x.insert(idx, v)
            
            self.assertEqual(find, v in list(range(5)))
            self.assertEqual(len(x), 6)
            self.assertTrue(all(x[i] <= x[i+1] for i in range(len(x)-1)))


class TestFull2Half(unittest.TestCase):
    def test_full2half(self):
        self.assertEqual(Full2Half.full2half("，；：？！"), ",;:?!")
        self.assertEqual(Full2Half.half2full(",;:?!"), "，；：？！")


class TestToken(unittest.TestCase):
    def test_assign_attr(self):
        tok = Token("-5.44", chunking='B-NP')
        self.assertTrue(hasattr(tok, 'chunking'))
        self.assertEqual(tok.chunking, 'B-NP')
        
    def test_adaptive_lower(self):
        tok = Token("Of")
        self.assertEqual(tok.text, "of")
        
        tok = Token("THE")
        self.assertEqual(tok.text, "the")
        
        tok = Token("marry")
        self.assertEqual(tok.text, "marry")
        
        tok = Token("MARRY")
        self.assertEqual(tok.text, "MARRY")
        
        tok = Token("WATERMELON")
        self.assertEqual(tok.text, "watermelon")
        
        tok = Token("WATERMELON-")
        self.assertEqual(tok.text, "WATERMELON-")
        
        tok = Token("Jack")
        self.assertEqual(tok.text, "jack")
        
        tok = Token("jACK")
        self.assertEqual(tok.text, "jACK")
        
    def test_en_shapes(self):
        tok = Token("Jack")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': True,
               'any_digit': False,
               'any_punct': False,
               'init_upper': True,
               'init_lower': False,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': False,
               'any_noninit_lower': True,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': True,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        self.assertTrue(all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names))
        
        
        tok = Token("INTRODUCTION")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': False,
               'any_digit': False,
               'any_punct': False,
               'init_upper': True,
               'init_lower': False,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': False,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': False,
               'typical_upper': True,
               'typical_lower': False,
               'apostrophe_end': False}
        self.assertTrue(all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names))
        
        
        tok = Token("is_1!!_IRR_demo")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': True,
               'any_digit': True,
               'any_punct': True,
               'init_upper': False,
               'init_lower': True,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': True,
               'any_noninit_digit': True,
               'any_noninit_punct': True,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        self.assertTrue(all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names))
        
        
        tok = Token("0是another@DEMO's")
        ans = {'any_ascii': True,
               'any_non_ascii': True,
               'any_upper': True,
               'any_lower': True,
               'any_digit': True,
               'any_punct': True,
               'init_upper': False,
               'init_lower': False,
               'init_digit': True,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': True,
               'any_noninit_digit': False,
               'any_noninit_punct': True,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': True}
        self.assertTrue(all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names))
        
        tok = Token("s")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': False,
               'any_lower': True,
               'any_digit': False,
               'any_punct': False,
               'init_upper': False,
               'init_lower': True,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': False,
               'any_noninit_lower': False,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        self.assertTrue(all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names))
        
        
    def test_numbers(self):
        tok = Token("5.44")
        self.assertEqual(tok.raw_text, "5.44")
        self.assertEqual(tok.text, '<real1>')
        self.assertTrue(tok.get_num_feature('<real1>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<real1>'))
        
        tok = Token("-5.44")
        self.assertEqual(tok.raw_text, "-5.44")
        self.assertEqual(tok.text, '<-real1>')
        self.assertTrue(tok.get_num_feature('<-real1>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-real1>'))
            
        tok = Token("4")
        self.assertEqual(tok.raw_text, "4")
        self.assertEqual(tok.text, "4")
        self.assertTrue(tok.get_num_feature('<int1>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int1>'))
        
        tok = Token("-4")
        self.assertEqual(tok.raw_text, "-4")
        self.assertEqual(tok.text, "-4")
        self.assertTrue(tok.get_num_feature('<-int1>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int1>'))
        
        tok = Token("511")
        self.assertEqual(tok.raw_text, "511")
        self.assertEqual(tok.text, "<int3>")
        self.assertTrue(tok.get_num_feature('<int3>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int3>'))
        
        tok = Token("-511")
        self.assertEqual(tok.raw_text, "-511")
        self.assertEqual(tok.text, "<-int3>")
        self.assertTrue(tok.get_num_feature('<-int3>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int3>'))
        
        tok = Token("2011")
        self.assertEqual(tok.raw_text, "2011")
        self.assertEqual(tok.text, "2011")
        self.assertTrue(tok.get_num_feature('<int4+>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int4+>'))
        
        tok = Token("-2011")
        self.assertEqual(tok.raw_text, "-2011")
        self.assertEqual(tok.text, "<-int4+>")
        self.assertTrue(tok.get_num_feature('<-int4+>'))
        self.assertFalse(any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int4+>'))
        
        
class TestTokenSequence(unittest.TestCase):
    def test_text(self):
        token_list = [Token(tok) for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        self.assertEqual(tokens.raw_text, ["This", "is", "a", "-3.14", "demo", "."])
        self.assertEqual(tokens.text, ["this", "is", "a", "<-real1>", "demo", "."])
        
    def test_ngrams(self):
        token_list = [Token(tok) for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        self.assertEqual(tokens.bigram, ["this-<sep>-is", "is-<sep>-a", "a-<sep>-<-real1>", 
                                         "<-real1>-<sep>-demo", "demo-<sep>-.", ".-<sep>-<pad>"])
        self.assertEqual(tokens.trigram, ["this-<sep>-is-<sep>-a", "is-<sep>-a-<sep>-<-real1>", 
                                          "a-<sep>-<-real1>-<sep>-demo", "<-real1>-<sep>-demo-<sep>-.", 
                                          "demo-<sep>-.-<sep>-<pad>", ".-<sep>-<pad>-<sep>-<pad>"])


class TestTags2Entities(unittest.TestCase):
    def test_tags2simple_entities(self):
        tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
        simple_entities = tags2simple_entities(tags, labeling='BIOES')
        self.assertEqual(len(simple_entities), 5)
        for ent in simple_entities:
            self.assertTrue(all(tag.split('-')[1] == ent['type'] for tag in tags[ent['start']:ent['stop']]))
            
        cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
        simple_entities = tags2simple_entities(cas_tags, labeling='BIOES')
        self.assertEqual(len(simple_entities), 5)
        for ent in simple_entities:
            self.assertEqual(ent['type'], '<pseudo-entity>')
            
    def test_tags2entities(self):
        nlp = spacy.load("en_core_web_sm")
        raw_text = "This is a -3.14 demo. Those are an APPLE and some glass bottles."
        tokens = build_token_sequence(raw_text, nlp)
        
        entities = [{'entity': 'demo', 'type': 'Ent', 'start': 16, 'end': 20},
                    {'entity': 'APPLE', 'type': 'Ent', 'start': 35, 'end': 40},
                    {'entity': 'glass bottles', 'type': 'Ent', 'start': 50, 'end': 63}]
        tags = ['O', 'O', 'O', 'O', 'S-Ent', 'O', 'O', 'O', 'O', 'S-Ent', 'O', 'O', 'B-Ent', 'E-Ent', 'O']
        
        tags_built, *_ = entities2tags(raw_text, tokens, entities, labeling='BIOES')
        entities_retr = tags2entities(raw_text, tokens, tags, labeling='BIOES')
        self.assertEqual(tags_built, tags)
        self.assertEqual(entities_retr, entities)
        
        entities_retr_spans = []
        for span in tokens.spans_within_max_length(10):
            entities_retr_spans.extend(tags2entities(raw_text, tokens[span], tags[span], labeling='BIOES'))
        self.assertEqual(entities_retr_spans, entities)
        
        
class TestTagHelper(unittest.TestCase):
    def test_dictionary(self):
        idx2tag = ['<pad>', 'O', 'B-A', 'I-A', 'E-A', 'S-A', 'B-B', 'I-B', 'E-B', 'S-B', 'B-C', 'I-C', 'E-C', 'S-C']
        tag2idx = {t: i for i, t in enumerate(idx2tag)}
        idx2cas_tag = ['<pad>', 'O', 'B', 'I', 'E', 'S']
        cas_tag2idx = {t: i for i, t in enumerate(idx2cas_tag)}
        idx2cas_type = ['<pad>', 'O', 'A', 'B', 'C']
        cas_type2idx = {t: i for i, t in enumerate(idx2cas_type)}
        tag_helper = TagHelper(cascade=False, labeling='BIOES')
        tag_helper.set_vocabs(idx2tag, tag2idx, idx2cas_tag, cas_tag2idx, idx2cas_type, cas_type2idx)
        
        tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
        tag_ids = [1, 1, 5, 6, 8, 1, 9, 1, 10, 11, 11, 12, 13, 0]
        self.assertEqual(tag_helper.tags2ids(tags), tag_ids)
        self.assertEqual(tag_helper.ids2tags(tag_ids), tags)
        self.assertEqual(tag_helper.modeling_tags2ids(tags), tag_ids)
        self.assertEqual(tag_helper.ids2modeling_tags(tag_ids), tags)
        
        tag_helper.cascade = True
        cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
        cas_tag_ids = [1, 1, 5, 2, 4, 1, 5, 1, 2, 3, 3, 4, 5, 0]
        self.assertEqual(tag_helper.cas_tags2ids(cas_tags), cas_tag_ids)
        self.assertEqual(tag_helper.ids2cas_tags(cas_tag_ids), cas_tags)
        self.assertEqual(tag_helper.modeling_tags2ids(cas_tags), cas_tag_ids)
        self.assertEqual(tag_helper.ids2modeling_tags(cas_tag_ids), cas_tags)
        
        cas_types = ['O', 'O', 'A', 'B', 'B', 'O', 'B', 'O', 'C', 'C', 'C', 'C', 'C', '<pad>']
        cas_type_ids = [1, 1, 2, 3, 3, 1, 3, 1, 4, 4, 4, 4, 4, 0]
        self.assertEqual(tag_helper.cas_types2ids(cas_types), cas_type_ids)
        self.assertEqual(tag_helper.ids2cas_types(cas_type_ids), cas_types)
        
    def test_cascade_transform(self):
        tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
        cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
        cas_types = ['O', 'O', 'A', 'B', 'B', 'O', 'B', 'O', 'C', 'C', 'C', 'C', 'C', '<pad>']
        cas_ent_slices = [slice(2, 3), slice(3, 5), slice(6, 7), slice(8, 12), slice(12, 13)]
        cas_ent_types = ['A', 'B', 'B', 'C', 'C']
        tag_helper = TagHelper(labeling='BIOES')
        
        self.assertEqual(tag_helper.build_cas_tags_by_tags(tags), cas_tags)
        self.assertEqual(tag_helper.build_cas_types_by_tags(tags), cas_types)
        self.assertEqual(tag_helper.build_cas_ent_slices_and_types_by_tags(tags)[0], cas_ent_slices)
        self.assertEqual(tag_helper.build_cas_ent_slices_and_types_by_tags(tags)[1], cas_ent_types)
        self.assertEqual(tag_helper.build_cas_ent_slices_by_cas_tags(cas_tags), cas_ent_slices)
        self.assertEqual(tag_helper.build_tags_by_cas_tags_and_types(cas_tags, cas_types), tags)
        self.assertEqual(tag_helper.build_tags_by_cas_tags_and_ent_slices_and_types(cas_tags, cas_ent_slices, cas_ent_types), tags)
        
        
def load_demo_data(labeling='BIOES', seed=515):
    with open(f"assets/data/covid19/{labeling}-train-demo-data-{seed}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(f"assets/data/covid19/{labeling}-val-demo-data-{seed}.pkl", 'rb') as f:
        val_data = pickle.load(f)
    with open(f"assets/data/covid19/{labeling}-test-demo-data-{seed}.pkl", 'rb') as f:
        test_data = pickle.load(f)
    return train_data, val_data, test_data

def build_demo_datasets(train_data, val_data, test_data, enum_fields=None, val_fields=None, cascade=False, labeling='BIOES'):
    train_set = COVID19Dataset(train_data, train=True,  labeled=True,  enum_fields=enum_fields, val_fields=val_fields, 
                               cascade=cascade, labeling=labeling)
    val_set   = COVID19Dataset(val_data,   train=False, labeled=True,  enum_fields=enum_fields, val_fields=val_fields,
                               cascade=cascade, labeling=labeling, vocabs=train_set.get_vocabs())
    test_set  = COVID19Dataset(test_data,  train=False, labeled=False, enum_fields=enum_fields, val_fields=val_fields,
                               cascade=cascade, labeling=labeling, vocabs=train_set.get_vocabs())
    return train_set, val_set, test_set


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_data, val_data, test_data = load_demo_data(labeling='BIOES', seed=515)
train_set, val_set, test_set = build_demo_datasets(train_data, val_data, test_data)
train_set_nofields, *_ = build_demo_datasets(train_data, val_data, test_data, enum_fields=[], val_fields=[])
train_set_cascade, *_ = build_demo_datasets(train_data, val_data, test_data, cascade=True)

train_data_BIO, val_data_BIO, test_data_BIO = load_demo_data(labeling='BIO', seed=515)
train_set_BIO, *_ = build_demo_datasets(train_data_BIO, val_data_BIO, test_data_BIO, labeling='BIO')

# https://nlp.stanford.edu/projects/glove/
glove = GloVe(name='6B', dim=100, root="assets/vector_cache", validate_file=False)


class TestBatching(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is available")
    def test_batch_to_cuda(self):
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                                  collate_fn=train_set.collate, pin_memory=True)
        for batch in train_loader:
            break
        
        self.assertTrue(batch.tok_ids.is_pinned())
        self.assertTrue(batch.enum['ent_tag'].is_pinned())
        self.assertTrue(batch.val['en_shape_features'].is_pinned())
        self.assertTrue(batch.seq_lens.is_pinned())
        self.assertTrue(batch.tags_objs[0].tag_ids.is_pinned())
        
        cpu = torch.device('cpu')
        gpu = torch.device('cuda:0')
        self.assertEqual(batch.tok_ids.device, cpu)
        batch = batch.to(gpu)
        self.assertEqual(batch.tok_ids.device, gpu)
        self.assertFalse(batch.tok_ids.is_pinned())
        
    def test_batches(self):
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                                  collate_fn=train_set.collate)
        val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, 
                                  collate_fn=val_set.collate)
        test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False, 
                                  collate_fn=test_set.collate)
        for batch in train_loader:
            break
        for batch in val_loader:
            break
        for batch in test_loader:
            break
        

class TestTagger(unittest.TestCase):
    def one_tagger_pass(self, tagger, train_set):
        count_trainable_params(tagger)
        tagger.eval()
        
        batch012 = train_set.collate([train_set[i] for i in range(0, 3)]).to(device)
        batch123 = train_set.collate([train_set[i] for i in range(1, 4)]).to(device)
        losses012, hidden012 = tagger(batch012, return_hidden=True)
        losses123, hidden123 = tagger(batch123, return_hidden=True)
        
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        self.assertTrue(delta_hidden.abs().max().item() < 1e-4)
        
        delta_losses = losses012[1:] - losses123[:-1]
        self.assertTrue(delta_losses.abs().max().item() < 1e-4)
        
        best_paths012 = tagger.decode(batch012)
        best_paths123 = tagger.decode(batch123)
        self.assertTrue(best_paths012[1:] == best_paths123[:-1])
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = NERTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
        
    def test_word_embedding_init(self):
        config, tag_helper = train_set.get_model_config()
        config = ConfigHelper.load_default_config(config, enc_arch='LSTM', dec_arch='CRF')
        tagger = build_tagger_by_config(config, tag_helper, train_set.tok_vocab.get_itos(), glove).to(device)
        self.one_tagger_pass(tagger, train_set)
        
        
    def test_train_steps(self):
        config, tag_helper = train_set.get_model_config()
        config = ConfigHelper.load_default_config(config, enc_arch='LSTM', dec_arch='CRF')
        tagger = build_tagger_by_config(config, tag_helper)
        batch = train_set.collate([train_set[i] for i in range(0, 4)]).to(device)
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = NERTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_steps(train_loader=[batch, batch], 
                            eval_loader=[batch, batch], 
                            n_epochs=10, disp_every_steps=2, eval_every_steps=6)
        
        
    def test_tagger_base(self):
        config, tag_helper = train_set.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set)
                
    def test_tagger_bert(self):
        tokenizer = BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-cased")
        bert = BertModel.from_pretrained("assets/transformers_cache/bert-base-cased").to(device)
        
        train_set_bert = COVID19Dataset(train_data, train=True,  labeled=True, bert_tokenizer=tokenizer,
                                        enum_fields=[], val_fields=[], 
                                        cascade=False, labeling='BIOES')
        config, tag_helper = train_set_bert.get_model_config()
        
        config = ConfigHelper.load_default_config(config, bert=bert, dec_arch='CRF')
        del config['tok'], config['char']
        config = ConfigHelper.update_full_dims(config)
        tagger = build_tagger_by_config(config, tag_helper, bert=bert).to(device)
        self.one_tagger_pass(tagger, train_set_bert)
        
        
    def test_tagger_shortcut(self):
        config, tag_helper = train_set.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                config['shortcut'] = True
                config = ConfigHelper.update_full_dims(config)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set)
                
    def test_tagger_cascade(self):
        config, tag_helper = train_set_cascade.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax-cascade', 'CRF-cascade']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_cascade)
                
    def test_tagger_BIO(self):
        config, tag_helper = train_set_BIO.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_BIO)
            
            
    def test_tagger_nofields(self):
        config, tag_helper = train_set_nofields.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_nofields)
            

class TestMLM(unittest.TestCase):
    def test_covid19_mlm(self):
        tokenizer = BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-cased")
        bert4mlm = BertForMaskedLM.from_pretrained("assets/transformers_cache/bert-base-cased").to(device)
        
        LABELING = 'BIOES'
        SEED = 515
        with open(f"assets/data/covid19/{LABELING}-train-demo-data-{SEED}.pkl", 'rb') as f:
            train_data = pickle.load(f)
            
        train_set = COVID19MLMDataset(train_data, tokenizer)
        batch = [train_set[i] for i in range(4)]
        batch012 = train_set.collate(batch[:3]).to(device)
        batch123 = train_set.collate(batch[1:]).to(device)
        
        loss012, MLM_scores012 = bert4mlm(input_ids=batch012.MLM_tok_ids, 
                                          attention_mask=(~batch012.attention_mask).type(torch.long), 
                                          labels=batch012.MLM_lab_ids)
        loss123, MLM_scores123 = bert4mlm(input_ids=batch123.MLM_tok_ids, 
                                          attention_mask=(~batch123.attention_mask).type(torch.long), 
                                          labels=batch123.MLM_lab_ids)
        
        min_step = min(MLM_scores012.size(1), MLM_scores123.size(1))
        delta_MLM_scores = MLM_scores012[1:, :min_step] - MLM_scores123[:-1, :min_step]
        self.assertTrue(delta_MLM_scores.abs().max().item() < 1e-4)
        
        optimizer = optim.AdamW(bert4mlm.parameters())
        trainer = MLMTrainer(bert4mlm, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
        
    def test_PMC_mlm(self):
        tokenizer = RobertaTokenizer.from_pretrained("assets/transformers_cache/roberta-base")
        roberta4mlm = RobertaForMaskedLM.from_pretrained("assets/transformers_cache/roberta-base").to(device)
        
        files = glob.glob("assets/data/PMC/comm_use/Cells/*.txt")
        train_set = PMCMLMDataset(files=files, tokenizer=tokenizer)
        train_loader = DataLoader(train_set, batch_size=4, 
                                  collate_fn=train_set.collate)
        for batch in train_loader:
            batch = batch.to(device)
            break
        
        loss012, MLM_scores012 = roberta4mlm(input_ids=batch.MLM_tok_ids[:3], 
                                             attention_mask=(~batch.attention_mask[:3]).type(torch.long), 
                                             labels=batch.MLM_lab_ids[:3])
        loss123, MLM_scores123 = roberta4mlm(input_ids=batch.MLM_tok_ids[1:], 
                                             attention_mask=(~batch.attention_mask[1:]).type(torch.long), 
                                             labels=batch.MLM_lab_ids[1:])
        
        min_step = min(MLM_scores012.size(1), MLM_scores123.size(1))
        delta_MLM_scores = MLM_scores012[1:, :min_step] - MLM_scores123[:-1, :min_step]
        self.assertTrue(delta_MLM_scores.abs().max().item() < 1e-4)
        
        optimizer = optim.AdamW(roberta4mlm.parameters(), lr=1e-4)
        trainer = MLMTrainer(roberta4mlm, optimizer=optimizer, device=device)
        trainer.train_epoch([batch])
        trainer.eval_epoch([batch])
        
        # trainer.train_steps(train_loader=[batch, batch], 
        #                     eval_loader=[batch, batch], 
        #                     n_epochs=10, disp_every_steps=2, eval_every_steps=6)
        
        
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
    
    
    
    