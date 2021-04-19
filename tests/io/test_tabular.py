# -*- coding: utf-8 -*-
import pytest
import jieba

from eznlp.io import TabularIO


class TestTabularIO(object):
    """
    References
    ----------
    [1] Zhang et al. 2015. Character-level convolutional networks for text classification. NIPS 2015.
    [2] Tang et al. 2015. Document modeling with gated recurrent neural network for sentiment classification. EMNLP 2015.
    [2] Chen et al. 2016. Neural sentiment classification with user and product attention. EMNLP 2016.
    """
    @pytest.mark.skip(reason="too slow and memory-consuming")
    def test_yelp_full(self, spacy_nlp_en):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep=",", mapping={"\\n": "\n", '\\"': '"'}, tokenize_callback=spacy_nlp_en, case_mode='lower')
        train_data = tabular_io.read("data/yelp_review_full/train.csv")
        test_data  = tabular_io.read("data/yelp_review_full/test.csv")
        
        assert len(train_data) == 650_000
        assert len(test_data) == 50_000
        
        
    @pytest.mark.skip(reason="too slow and memory-consuming")
    def test_yelp_polarity(self, spacy_nlp_en):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep=",", mapping={"\\n": "\n", '\\"': '"'}, tokenize_callback=spacy_nlp_en, case_mode='lower')
        train_data = tabular_io.read("data/yelp_review_polarity/train.csv")
        test_data  = tabular_io.read("data/yelp_review_polarity/test.csv")
        
        assert len(train_data) == 560_000
        assert len(test_data) == 38_000
        
        
    @pytest.mark.slow
    def test_imdb_with_up(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/imdb.train.txt.ss")
        dev_data   = tabular_io.read("data/Tang2015/imdb.dev.txt.ss")
        test_data  = tabular_io.read("data/Tang2015/imdb.test.txt.ss")
        
        assert len(train_data) == 67_426
        assert len(dev_data) == 8_381
        assert len(test_data) == 9_112
        assert len(train_data) + len(dev_data) + len(test_data) == 84_919
        
        
    @pytest.mark.slow
    def test_yelp2013(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.train.ss")
        dev_data   = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.dev.ss")
        test_data  = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss")
        
        assert len(train_data) == 62_522
        assert len(dev_data) == 7_773
        assert len(test_data) == 8_671
        assert len(train_data) + len(dev_data) + len(test_data) == 78_966
        
        
    @pytest.mark.slow
    def test_yelp2014(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.train.ss")
        dev_data   = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.dev.ss")
        test_data  = tabular_io.read("data/Tang2015/yelp-2014-seg-20-20.test.ss")
        
        assert len(train_data) == 183_019
        assert len(dev_data) == 22_745
        assert len(test_data) == 25_399
        assert len(train_data) + len(dev_data) + len(test_data) == 231_163
        
        
    @pytest.mark.slow
    def test_Ifeng(self):
        tabular_io = TabularIO(text_col_id=2, label_col_id=0, sep=',', tokenize_callback=jieba.cut, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/Ifeng/train.csv")
        test_data  = tabular_io.read("data/Ifeng/test.csv")
        
        assert len(train_data) == 800_000
        assert len(test_data) == 50_000
        
        
    def test_ChnSentiCorp(self):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep='\t', header=0, tokenize_callback=jieba.cut, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/ChnSentiCorp/train.tsv")
        dev_data   = tabular_io.read("data/ChnSentiCorp/dev.tsv")
        test_data  = tabular_io.read("data/ChnSentiCorp/test.tsv")
        
        assert len(train_data) == 9_146
        assert len(dev_data) == 1_200
        assert len(test_data) == 1_200
        
        
    @pytest.mark.slow
    def test_THUCNews_10(self):
        tabular_io = TabularIO(text_col_id=1, label_col_id=0, sep='\t', tokenize_callback=jieba.cut, encoding='utf-8', case_mode='lower')
        train_data = tabular_io.read("data/THUCNews-10/cnews.train.txt")
        dev_data   = tabular_io.read("data/THUCNews-10/cnews.val.txt")
        test_data  = tabular_io.read("data/THUCNews-10/cnews.test.txt")
        
        assert len(train_data) == 50_000
        assert len(dev_data) == 5_000
        assert len(test_data) == 10_000
        