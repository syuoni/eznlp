# -*- coding: utf-8 -*-
from eznlp.io import Src2TrgIO


class TestSrc2TrgIO(object):
    def test_multi30k(self, spacy_nlp_en, spacy_nlp_de):
        io = Src2TrgIO(tokenize_callback=spacy_nlp_de, trg_tokenize_callback=spacy_nlp_en, encoding='utf-8', case_mode='Lower', number_mode='None')
        train_data = io.read("data/multi30k/train.en", "data/multi30k/train.de")
        dev_data   = io.read("data/multi30k/val.en", "data/multi30k/val.de")
        test_data  = io.read("data/multi30k/test2016.en", "data/multi30k/test2016.de")
        
        assert len(train_data) == 29_000
        assert len(dev_data) == 1_014
        assert len(test_data) == 1_000
        
        
    def test_iwslt14_en_de(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        train_data = io.read("data/iwslt14.tokenized.de-en/train.en", "data/iwslt14.tokenized.de-en/train.de")
        dev_data   = io.read("data/iwslt14.tokenized.de-en/valid.en", "data/iwslt14.tokenized.de-en/valid.de")
        test_data  = io.read("data/iwslt14.tokenized.de-en/test.en", "data/iwslt14.tokenized.de-en/test.de")
        
        assert len(train_data) == 160_239
        assert len(dev_data) == 7_283
        assert len(test_data) == 6_750
        
        
    def test_wmt14_en_de(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt17_en_de/train.en", "data/wmt17_en_de/train.de")
        dev_data   = io.read("data/wmt17_en_de/valid.en", "data/wmt17_en_de/valid.de")
        test_data  = io.read("data/wmt17_en_de/test.en", "data/wmt17_en_de/test.de")
        
        # assert len(train_data) == 3_961_179
        assert len(dev_data) == 40_058
        assert len(test_data) == 3_003
        
        
    def test_wmt14_en_fr(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt14_en_fr/train.en", "data/wmt14_en_fr/train.fr")
        dev_data   = io.read("data/wmt14_en_fr/valid.en", "data/wmt14_en_fr/valid.fr")
        test_data  = io.read("data/wmt14_en_fr/test.en", "data/wmt14_en_fr/test.fr")
        
        # assert len(train_data) == 35_762_532
        assert len(dev_data) == 26_854
        assert len(test_data) == 3_003
        
        
    def test_wmt14_en_de_from_torchtext(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt14/train.tok.clean.bpe.32000.en", "data/wmt14/train.tok.clean.bpe.32000.de")
        dev_data   = io.read("data/wmt14/newstest2013.tok.bpe.32000.en", "data/wmt14/newstest2013.tok.bpe.32000.de")
        test_data  = io.read("data/wmt14/newstest2014.tok.bpe.32000.en", "data/wmt14/newstest2014.tok.bpe.32000.de")
        
        # assert len(train_data) == 4_500_966
        assert len(dev_data) == 3_000
        assert len(test_data) == 3_003
