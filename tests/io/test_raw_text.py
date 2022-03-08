# -*- coding: utf-8 -*-
import jieba
import transformers

from eznlp.io import RawTextIO


class TestRawTextIO(object):
    def test_yelp_full(self):
        tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers/bert-base-chinese")
        io = RawTextIO(tokenizer.tokenize, jieba.tokenize, max_len=510, document_sep_starts=["-DOCSTART-", "<doc", "</doc"], encoding='utf-8')
        data = io.read("data/Wikipedia/text-zh/AA/wiki_00")
        io.write(data, "data/Wikipedia/text-zh/AA/wiki_00.cache")
        
        io = RawTextIO(encoding='utf-8')
        reloaded = io.read("data/Wikipedia/text-zh/AA/wiki_00.cache")
        assert reloaded == data
