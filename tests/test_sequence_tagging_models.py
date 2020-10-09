# -*- coding: utf-8 -*-
import pytest
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.experimental.vectors import GloVe
from transformers import BertTokenizer, BertModel

from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import ConfigHelper
from eznlp.sequence_tagging import build_tagger_by_config
from eznlp.sequence_tagging import NERTrainer


def load_demo_data(labeling='BIOES', seed=515):
    with open(f"assets/data/covid19/{labeling}-train-demo-data-{seed}.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(f"assets/data/covid19/{labeling}-val-demo-data-{seed}.pkl", 'rb') as f:
        val_data = pickle.load(f)
    with open(f"assets/data/covid19/{labeling}-test-demo-data-{seed}.pkl", 'rb') as f:
        test_data = pickle.load(f)
    return train_data, val_data, test_data


def build_demo_datasets(train_data, val_data, test_data, enum_fields=None, val_fields=None, 
                        cascade=False, labeling='BIOES'):
    train_set = SequenceTaggingDataset(train_data, enum_fields=enum_fields, val_fields=val_fields, 
                                       cascade=cascade, labeling=labeling)
    val_set   = SequenceTaggingDataset(val_data,   enum_fields=enum_fields, val_fields=val_fields,
                                       cascade=cascade, labeling=labeling, vocabs=train_set.get_vocabs())
    test_set  = SequenceTaggingDataset(test_data,  enum_fields=enum_fields, val_fields=val_fields,
                                       cascade=cascade, labeling=labeling, vocabs=train_set.get_vocabs())
    return train_set, val_set, test_set

@pytest.fixture
def glove100():
    # https://nlp.stanford.edu/projects/glove/
    return GloVe(name='6B', dim=100, root="assets/vector_cache", validate_file=False)

@pytest.fixture
def BIOES_data():
    return load_demo_data(labeling='BIOES', seed=515)

@pytest.fixture
def BIOES_datasets(BIOES_data):
    return build_demo_datasets(*BIOES_data)
    
@pytest.fixture
def BIOES_datasets_nofields(BIOES_data):
    return build_demo_datasets(*BIOES_data, enum_fields=[], val_fields=[])

@pytest.fixture
def BIOES_datasets_morefields(BIOES_data):
    return build_demo_datasets(*BIOES_data, 
                               enum_fields=SequenceTaggingDataset._pre_enum_fields+['upos', 'detailed_pos', 'dep', 'ent_tag'], 
                               val_fields=SequenceTaggingDataset._pre_val_fields+['covid19tag'])

@pytest.fixture
def BIOES_datasets_cascade(BIOES_data):
    return build_demo_datasets(*BIOES_data, cascade=True)

@pytest.fixture
def BIO_data():
    return load_demo_data(labeling='BIO', seed=515)

@pytest.fixture
def BIO_datasets(BIO_data):
    return build_demo_datasets(*BIO_data, labeling='BIO')

@pytest.fixture
def BERT_with_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-cased")
    bert = BertModel.from_pretrained("assets/transformers_cache/bert-base-cased")
    return bert, tokenizer


class TestBatching(object):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA available")
    def test_batch_to_cuda(self, BIOES_datasets, device):
        train_set, val_set, test_set = BIOES_datasets
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                                  collate_fn=train_set.collate, pin_memory=True)
        for batch in train_loader:
            break
        
        assert batch.tok_ids.is_pinned()
        assert batch.enum['ent_tag'].is_pinned()
        assert batch.val['en_shape_features'].is_pinned()
        assert batch.seq_lens.is_pinned()
        assert batch.tags_objs[0].tag_ids.is_pinned()
        
        cpu = torch.device('cpu')
        gpu = torch.device('cuda:0')
        assert batch.tok_ids.device == cpu
        batch = batch.to(gpu)
        assert batch.tok_ids.device == gpu
        assert not batch.tok_ids.is_pinned()
        
        
    def test_batches(self, BIOES_datasets):
        train_set, val_set, test_set = BIOES_datasets
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
        
        
class TestTagger(object):
    def one_tagger_pass(self, tagger, train_set, device):
        tagger.eval()
        
        batch012 = train_set.collate([train_set[i] for i in range(0, 3)]).to(device)
        batch123 = train_set.collate([train_set[i] for i in range(1, 4)]).to(device)
        losses012, hidden012 = tagger(batch012, return_hidden=True)
        losses123, hidden123 = tagger(batch123, return_hidden=True)
        
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        assert delta_hidden.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 1e-4
        
        best_paths012 = tagger.decode(batch012)
        best_paths123 = tagger.decode(batch123)
        assert best_paths012[1:] == best_paths123[:-1]
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = NERTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
        
    def test_word_embedding_init(self, BIOES_datasets, glove100, device):
        train_set, val_set, test_set = BIOES_datasets
        config, tag_helper = train_set.get_model_config()
        config = ConfigHelper.load_default_config(config, enc_arch='LSTM', dec_arch='CRF')
        tagger = build_tagger_by_config(config, tag_helper, train_set.tok_vocab.get_itos(), glove100).to(device)
        self.one_tagger_pass(tagger, train_set, device)
        
        
    def test_train_steps(self, BIOES_datasets, device):
        train_set, val_set, test_set = BIOES_datasets
        config, tag_helper = train_set.get_model_config()
        config = ConfigHelper.load_default_config(config, enc_arch='LSTM', dec_arch='CRF')
        tagger = build_tagger_by_config(config, tag_helper)
        batch = train_set.collate([train_set[i] for i in range(0, 4)]).to(device)
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = NERTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_steps(train_loader=[batch, batch], 
                            eval_loader=[batch, batch], 
                            n_epochs=10, disp_every_steps=2, eval_every_steps=6)
        
        
    def test_tagger_base(self, BIOES_datasets, device):
        train_set, val_set, test_set = BIOES_datasets
        config, tag_helper = train_set.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set, device)
                
                
    def test_tagger_bert(self, BIOES_data, BERT_with_tokenizer, device):
        train_data, *_ = BIOES_data
        bert, tokenizer = BERT_with_tokenizer
        
        train_set_bert = SequenceTaggingDataset(train_data, enum_fields=[], val_fields=[], 
                                                sub_tokenizer=tokenizer, cascade=False, labeling='BIOES')
        config, tag_helper = train_set_bert.get_model_config()
        config = ConfigHelper.load_default_config(config, bert=bert, dec_arch='CRF')
        del config['tok'], config['char']
        config = ConfigHelper.update_full_dims(config)
        tagger = build_tagger_by_config(config, tag_helper, bert=bert).to(device)
        self.one_tagger_pass(tagger, train_set_bert, device)
        
        
    def test_tagger_shortcut(self, BIOES_datasets, device):
        train_set, val_set, test_set = BIOES_datasets
        config, tag_helper = train_set.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                config['shortcut'] = True
                config = ConfigHelper.update_full_dims(config)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set, device)
                
                
    def test_tagger_cascade(self, BIOES_datasets_cascade, device):
        train_set_cascade, *_ = BIOES_datasets_cascade
        config, tag_helper = train_set_cascade.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax-cascade', 'CRF-cascade']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_cascade, device)
            
            
    def test_tagger_nofields(self, BIOES_datasets_nofields, device):
        train_set_nofields, *_ = BIOES_datasets_nofields
        config, tag_helper = train_set_nofields.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_nofields, device)
                
                
    def test_tagger_morefields(self, BIOES_datasets_morefields, device):
        train_set_morefields, *_ = BIOES_datasets_morefields
        config, tag_helper = train_set_morefields.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_morefields, device)
                

    def test_tagger_BIO(self, BIO_datasets, device):
        train_set_BIO, *_ = BIO_datasets
        config, tag_helper = train_set_BIO.get_model_config()
        for enc_arch in ['LSTM', 'CNN', 'Transformer']:
            for dec_arch in ['softmax', 'CRF']:
                config = ConfigHelper.load_default_config(config, enc_arch=enc_arch, dec_arch=dec_arch)
                tagger = build_tagger_by_config(config, tag_helper).to(device)
                self.one_tagger_pass(tagger, train_set_BIO, device)
                
                