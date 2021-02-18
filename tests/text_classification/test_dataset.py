# -*- coding: utf-8 -*-
import pytest
import copy

import pandas as pd
from eznlp.pretrained import BertLikeConfig
from eznlp.text_classification import TextClassifierConfig, TextClassificationDataset
from eznlp.text_classification.io import TabularIO


class TestTextClassificationDataset(object):
    @pytest.mark.slow
    def test_pre_truncation(self, bert_with_tokenizer):
        bert, tokenizer = copy.deepcopy(bert_with_tokenizer)
        config = TextClassifierConfig(ohots=None, 
                                      encoder=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert))
        
        tabular_io = TabularIO(text_col_id=3, label_col_id=2)
        dev_data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
        dev_data = [data_entry for data_entry in dev_data if len(data_entry['tokens']) >= 500]
        assert max(len(data_entry['tokens']) for data_entry in dev_data) > 510
        
        dev_set = TextClassificationDataset(dev_data, config)
        dev_set._truncate_tokens()
        sub_lens = [sum(len(tokenizer.tokenize(word)) for word in data_entry['tokens'].raw_text) for data_entry in dev_set.data]
        sub_lens = pd.Series(sub_lens)
        
        assert (sub_lens <= 510).all()
        assert (sub_lens == 510).sum() >= (len(sub_lens) / 2)
        
        