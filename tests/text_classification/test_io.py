# -*- coding: utf-8 -*-
from eznlp.text_classification.io import TabularIO


class TestTabularIO(object):
    """
    References
    ----------
    [1] Chen et al. 2016. Neural sentiment classification with user and product attention. 
    """
    def test_yelp2013(self):
        tabular_io = TabularIO(text_col_id=3, label_col_id=2)
        # train_data = tabular_io.read("assets/data/Tang2015/yelp-2013-seg-20-20.train.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
        val_data   = tabular_io.read("assets/data/Tang2015/yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
        test_data  = tabular_io.read("assets/data/Tang2015/yelp-2013-seg-20-20.test.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
        
        # 78_966
        # assert len(train_data) == 62_522
        assert len(val_data) == 7_773
        assert len(test_data) == 8_671
        
        