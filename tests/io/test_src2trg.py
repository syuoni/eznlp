# -*- coding: utf-8 -*-
from eznlp.io import Src2TrgIO


class TestSrc2TrgIO(object):
    def test_multi30k(self, spacy_nlp_en, spacy_nlp_de):
        io = Src2TrgIO(tokenize_callback=spacy_nlp_de, trg_tokenize_callback=spacy_nlp_en, encoding='utf-8', case_mode='Lower', number_mode='None')
        train_data = io.read("data/multi30k/train.de", "data/multi30k/train.en")
        dev_data   = io.read("data/multi30k/val.de", "data/multi30k/val.en")
        test_data  = io.read("data/multi30k/test2016.de", "data/multi30k/test2016.en")
        
        assert len(train_data) == 29_000
        assert len(dev_data) == 1_014
        assert len(test_data) == 1_000
        
        
    def test_iwslt14_en_de(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        train_data = io.read("data/iwslt14.tokenized.de-en/train.de", "data/iwslt14.tokenized.de-en/train.en")
        dev_data   = io.read("data/iwslt14.tokenized.de-en/valid.de", "data/iwslt14.tokenized.de-en/valid.en")
        test_data  = io.read("data/iwslt14.tokenized.de-en/test.de", "data/iwslt14.tokenized.de-en/test.en")
        
        assert len(train_data) == 160_239
        assert len(dev_data) == 7_283
        assert len(test_data) == 6_750
        
        
    def test_wmt14_en_de(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt14/train.tok.clean.bpe.32000.de", "data/wmt14/train.tok.clean.bpe.32000.en")
        dev_data   = io.read("data/wmt14/newstest2013.tok.bpe.32000.de", "data/wmt14/newstest2013.tok.bpe.32000.en")
        test_data  = io.read("data/wmt14/newstest2014.tok.bpe.32000.de", "data/wmt14/newstest2014.tok.bpe.32000.en")
        
        # assert len(train_data) == 4_500_966
        assert len(dev_data) == 3_000
        assert len(test_data) == 3_003
        
        
    def test_wmt14_en_fr(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt14_en_fr/train.fr", "data/wmt14_en_fr/train.en")
        dev_data   = io.read("data/wmt14_en_fr/valid.fr", "data/wmt14_en_fr/valid.en")
        test_data  = io.read("data/wmt14_en_fr/test.fr", "data/wmt14_en_fr/test.en")
        
        # assert len(train_data) == 35_762_532
        assert len(dev_data) == 26_854
        assert len(test_data) == 3_003
