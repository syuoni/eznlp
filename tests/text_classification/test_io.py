# -*- coding: utf-8 -*-
import pytest

from eznlp.text_classification.io import TabularIO


class TestTabularIO(object):
    """
    References
    ----------
    [1] Zhang et al. 2015. Character-level convolutional networks for text classification. NIPS 2015.
    [2] Tang et al. 2015. Document modeling with gated recurrent neural network for sentiment classification. EMNLP 2015.
    [2] Chen et al. 2016. Neural sentiment classification with user and product attention. EMNLP 2016.
    """
    @pytest.mark.slow
    def test_yelp_full(self, spacy_nlp_en):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, mapping={"\\n": "\n", '\\"': '"'}, tokenize_callback=spacy_nlp_en, case_mode='lower')
        train_data = tabular_io.read("data/yelp_review_full/train.csv", sep=",")
        test_data  = tabular_io.read("data/yelp_review_full/test.csv", sep=",")
        
        assert len(train_data) == 650_000
        assert len(test_data) == 50_000
        
        
    @pytest.mark.slow
    def test_yelp_polarity(self, spacy_nlp_en):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, mapping={"\\n": "\n", '\\"': '"'}, tokenize_callback=spacy_nlp_en, case_mode='lower')
        train_data = tabular_io.read("data/yelp_review_polarity/train.csv", sep=",")
        test_data  = tabular_io.read("data/yelp_review_polarity/test.csv", sep=",")
        
        assert len(train_data) == 560_000
        assert len(test_data) == 38_000
        
        
    @pytest.mark.slow
    def test_imdb(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, mapping={"<sssss>": "\n"}, case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/imdb.train.txt.ss", encoding='utf-8', sep="\t\t")
        dev_data   = tabular_io.read("data/Tang2015/imdb.dev.txt.ss", encoding='utf-8', sep="\t\t")
        test_data  = tabular_io.read("data/Tang2015/imdb.test.txt.ss", encoding='utf-8', sep="\t\t")
        
        assert len(train_data) == 67_426
        assert len(dev_data) == 8_381
        assert len(test_data) == 9_112
        assert len(train_data) + len(dev_data) + len(test_data) == 84_919
        
        
    @pytest.mark.slow
    def test_yelp2013(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, mapping={"<sssss>": "\n"}, case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.train.ss", encoding='utf-8', sep="\t\t")
        dev_data   = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t")
        test_data  = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss", encoding='utf-8', sep="\t\t")
        
        assert len(train_data) == 62_522
        assert len(dev_data) == 7_773
        assert len(test_data) == 8_671
        assert len(train_data) + len(dev_data) + len(test_data) == 78_966
        
        
    @pytest.mark.slow
    def test_yelp2014(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, mapping={"<sssss>": "\n"}, case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.train.ss", encoding='utf-8', sep="\t\t")
        dev_data   = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t")
        test_data  = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.test.ss", encoding='utf-8', sep="\t\t")
        
        assert len(train_data) == 183_019
        assert len(dev_data) == 22_745
        assert len(test_data) == 25_399
        assert len(train_data) + len(dev_data) + len(test_data) == 231_163
        
        