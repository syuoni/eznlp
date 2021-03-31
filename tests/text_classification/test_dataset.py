# -*- coding: utf-8 -*-
import pandas

from eznlp.pretrained import BertLikeConfig
from eznlp.text_classification import TextClassifierConfig, TextClassificationDataset
from eznlp.text_classification.io import TabularIO


class TestTextClassificationDataset(object):
    def test_truncation(self, bert_with_tokenizer):
        bert, tokenizer = bert_with_tokenizer
        config = TextClassifierConfig(ohots=None, 
                                      encoder=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert))
        
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, mapping={"<sssss>": "\n"}, case_mode='lower')
        data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss", encoding='utf-8', sep="\t\t")
        data = [data_entry for data_entry in data if len(data_entry['tokens']) >= 500]
        assert max(len(data_entry['tokens']) for data_entry in data) > 510
        
        dataset = TextClassificationDataset(data, config)
        dataset.truncate_for_bert_like()
        sub_lens = [sum(len(tokenizer.tokenize(word)) for word in data_entry['tokens'].raw_text) for data_entry in dataset.data]
        sub_lens = pandas.Series(sub_lens)
        
        assert (sub_lens <= 510).all()
        assert (sub_lens == 510).sum() >= (len(sub_lens) / 2)
        
        