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
        
        
    def test_wmt14(self):
        io = Src2TrgIO(tokenize_callback=None, trg_tokenize_callback=None, encoding='utf-8', case_mode='Lower', number_mode='None')
        # train_data = io.read("data/wmt14/train.tok.clean.bpe.32000.de", "data/wmt14/train.tok.clean.bpe.32000.en")
        dev_data   = io.read("data/wmt14/newstest2013.tok.bpe.32000.de", "data/wmt14/newstest2013.tok.bpe.32000.en")
        test_data  = io.read("data/wmt14/newstest2014.tok.bpe.32000.de", "data/wmt14/newstest2014.tok.bpe.32000.en")
        
        # assert len(train_data) == 4_500_966
        assert len(dev_data) == 3_000
        assert len(test_data) == 3_003
