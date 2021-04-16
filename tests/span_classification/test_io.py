# -*- coding: utf-8 -*-
from eznlp.span_classification.io import JsonIO


class TestTabularIO(object):
    """
    References
    ----------
    [1] Eberts and Ulges. 2019. Span-based joint entity and relation extraction with Transformer pre-training. ECAI 2020.
    """
    def test_conll2004(self):
        json_io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        train_data = json_io.read("data/conll2004/conll04_train.json")
        dev_data   = json_io.read("data/conll2004/conll04_dev.json")
        test_data  = json_io.read("data/conll2004/conll04_test.json")
        
        assert len(train_data) == 922
        assert len(dev_data) == 231
        assert len(test_data) == 288
        
        
    def test_SciERC(self):
        json_io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        train_data = json_io.read("data/SciERC/scierc_train.json")
        dev_data   = json_io.read("data/SciERC/scierc_dev.json")
        test_data  = json_io.read("data/SciERC/scierc_test.json")
        
        assert len(train_data) == 1_861
        assert len(dev_data) == 275
        assert len(test_data) == 551
        
        
    def test_ADE(self):
        json_io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        data = json_io.read("data/ADE/ade_full.json")
        
        assert len(data) == 4_272
        assert sum(len(ex['chunks']) for ex in data) == 10_839
        