# -*- coding: utf-8 -*-
import pytest

from eznlp.io import CategoryFolderIO


class TestCategoryFolderIO(object):
    """
    References
    ----------
    [1] Zhang et al. 2015. Character-level convolutional networks for text classification. NIPS 2015.
    """
    @pytest.mark.slow
    def test_imdb(self, spacy_nlp_en):
        folder_io = CategoryFolderIO(categories=["pos", "neg"], mapping={"<br />": "\n"}, tokenize_callback=spacy_nlp_en, encoding='utf-8', case_mode='lower')
        train_data = folder_io.read("data/imdb/train")
        test_data  = folder_io.read("data/imdb/test")
        
        assert len(train_data) == 25_000
        assert len(test_data) == 25_000
        