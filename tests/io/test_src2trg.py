# -*- coding: utf-8 -*-
from eznlp.io import Src2TrgIO


class TestSrc2TrgIO(object):
    def test_multi30k(self, spacy_nlp_en, spacy_nlp_de):
        io = Src2TrgIO(tokenize_callback=spacy_nlp_de, trg_tokenize_callback=spacy_nlp_en, encoding='utf-8', case_mode='lower', number_mode='None')
        train_data = io.read("data/multi30k/train.de", "data/multi30k/train.en")
        dev_data   = io.read("data/multi30k/val.de", "data/multi30k/val.en")
        test_data  = io.read("data/multi30k/test2016.de", "data/multi30k/test2016.en")
        
        assert len(train_data) == 29_000
        assert len(dev_data) == 1_014
        assert len(test_data) == 1_000
