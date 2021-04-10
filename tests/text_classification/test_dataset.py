# -*- coding: utf-8 -*-
import pytest
import pandas

from eznlp.pretrained.bert_like import truncate_for_bert_like, _tokenized2nested
from eznlp.text_classification.io import TabularIO


class TestTextClassificationDataset(object):
    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["head+tail", "head-only", "tail-only"])
    def test_truncation(self, mode, bert_with_tokenizer):
        bert, tokenizer = bert_with_tokenizer
        max_len = tokenizer.model_max_length - 2
        
        tabular_io = TabularIO(text_col_id=3, label_col_id=2, mapping={"<sssss>": "\n"}, case_mode='lower')
        data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss", encoding='utf-8', sep="\t\t")
        data = [data_entry for data_entry in data if len(data_entry['tokens']) >= max_len-10]
        assert max(len(data_entry['tokens']) for data_entry in data) > max_len
        
        data = truncate_for_bert_like(data, tokenizer, mode)
        sub_lens = [sum(len(word) for word in _tokenized2nested(data_entry['tokens'].raw_text, tokenizer)) for data_entry in data]
        sub_lens = pandas.Series(sub_lens)
        
        assert (sub_lens <= max_len).all()
        assert (sub_lens == max_len).sum() >= (len(sub_lens) / 2)
        
        