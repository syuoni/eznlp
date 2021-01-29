# -*- coding: utf-8 -*-
import pytest

import pandas as pd
from eznlp.encoder import PreTrainedEmbedderConfig
from eznlp.text_classification import TextClassifierConfig, TextClassificationDataset
from eznlp.text_classification.io import TabularIO


class TestTextClassificationDataset(object):
    @pytest.mark.slow
    def test_pre_truncation(self, bert_with_tokenizer):
        bert, tokenizer = bert_with_tokenizer
        bert_like_embedder_config = PreTrainedEmbedderConfig(arch='BERT', 
                                                             out_dim=bert.config.hidden_size, 
                                                             tokenizer=tokenizer, 
                                                             from_tokenized=True, 
                                                             pre_truncation=True)
        config = TextClassifierConfig(encoder=None, bert_like_embedder=bert_like_embedder_config)
        
        tabular_io = TabularIO(text_col_id=3, label_col_id=2)
        dev_data   = tabular_io.read("assets/data/Tang2015/yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
        dev_data = [curr_data for curr_data in dev_data if len(curr_data['tokens']) >= 500]
        assert max(len(curr_data['tokens']) for curr_data in dev_data) > 510
        
        dev_set = TextClassificationDataset(dev_data, config)
        sub_lens = [sum(len(tokenizer.tokenize(word)) for word in curr_data['tokens'].raw_text) for curr_data in dev_set.data]
        sub_lens = pd.Series(sub_lens)
        
        assert (sub_lens <= 510).all()
        assert (sub_lens == 510).sum() >= (len(sub_lens) / 2)
        
        