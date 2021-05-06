# -*- coding: utf-8 -*-
from eznlp.io import JsonIO


class TestJsonIO(object):
    """
    References
    ----------
    [1] Eberts and Ulges. 2019. Span-based joint entity and relation extraction with Transformer pre-training. ECAI 2020.
    """
    def test_conll2004(self):
        json_io = JsonIO(text_key='tokens', 
                         chunk_key='entities', 
                         chunk_type_key='type', 
                         chunk_start_key='start', 
                         chunk_end_key='end', 
                         relation_key='relations', 
                         relation_type_key='type', 
                         relation_head_key='head', 
                         relation_tail_key='tail')
        train_data = json_io.read("data/conll2004/conll04_train.json")
        dev_data   = json_io.read("data/conll2004/conll04_dev.json")
        test_data  = json_io.read("data/conll2004/conll04_test.json")
        
        assert len(train_data) == 922
        assert sum(len(ex['chunks']) for ex in train_data) == 3_377
        assert sum(len(ex['relations']) for ex in train_data) == 1_283
        assert len(dev_data) == 231
        assert sum(len(ex['chunks']) for ex in dev_data) == 893
        assert sum(len(ex['relations']) for ex in dev_data) == 343
        assert len(test_data) == 288
        assert sum(len(ex['chunks']) for ex in test_data) == 1_079
        assert sum(len(ex['relations']) for ex in test_data) == 422
        
        
    def test_SciERC(self):
        json_io = JsonIO(text_key='tokens', 
                         chunk_key='entities', 
                         chunk_type_key='type', 
                         chunk_start_key='start', 
                         chunk_end_key='end', 
                         relation_key='relations', 
                         relation_type_key='type', 
                         relation_head_key='head', 
                         relation_tail_key='tail')
        train_data = json_io.read("data/SciERC/scierc_train.json")
        dev_data   = json_io.read("data/SciERC/scierc_dev.json")
        test_data  = json_io.read("data/SciERC/scierc_test.json")
        
        assert len(train_data) == 1_861
        assert sum(len(ex['chunks']) for ex in train_data) == 5_598
        assert sum(len(ex['relations']) for ex in train_data) == 3_219
        assert len(dev_data) == 275
        assert sum(len(ex['chunks']) for ex in dev_data) == 811
        assert sum(len(ex['relations']) for ex in dev_data) == 455
        assert len(test_data) == 551
        assert sum(len(ex['chunks']) for ex in test_data) == 1_685
        assert sum(len(ex['relations']) for ex in test_data) == 974
        
        
    def test_ADE(self):
        json_io = JsonIO(text_key='tokens', 
                         chunk_key='entities', 
                         chunk_type_key='type', 
                         chunk_start_key='start', 
                         chunk_end_key='end', 
                         relation_key='relations', 
                         relation_type_key='type', 
                         relation_head_key='head', 
                         relation_tail_key='tail')
        data = json_io.read("data/ADE/ade_full.json")
        
        assert len(data) == 4_272
        assert sum(len(ex['chunks']) for ex in data) == 10_839
        assert sum(len(ex['relations']) for ex in data) == 6_821
        
        