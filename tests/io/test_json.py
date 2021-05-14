# -*- coding: utf-8 -*-
from eznlp.io import JsonIO, SQuADIO
from eznlp.utils.chunk import detect_nested, filter_clashed_by_priority


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
        
        assert not any(detect_nested(ex['chunks']) for data in [train_data, dev_data, test_data] for ex in data)
        assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=False) == ex['chunks'] for data in [train_data, dev_data, test_data] for ex in data)
        
        
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
        assert sum(len(ex['relations']) for ex in train_data) == 3_215  # 4 duplicated relations dropped here
        assert len(dev_data) == 275
        assert sum(len(ex['chunks']) for ex in dev_data) == 811
        assert sum(len(ex['relations']) for ex in dev_data) == 455
        assert len(test_data) == 551
        assert sum(len(ex['chunks']) for ex in test_data) == 1_685
        assert sum(len(ex['relations']) for ex in test_data) == 974
        
        assert any(detect_nested(ex['chunks']) for data in [train_data, dev_data, test_data] for ex in data)
        assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=True) == ex['chunks'] for data in [train_data, dev_data, test_data] for ex in data)
        
        
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
        
        assert any(detect_nested(ex['chunks']) for ex in data)
        assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=True) == ex['chunks'] for ex in data)



class TestSQuADIO(object):
    def test_squad_v2(self, spacy_nlp_en):
        io = SQuADIO(tokenize_callback=spacy_nlp_en, verbose=False)
        train_data, train_errors, train_mismatches = io.read("data/SQuAD/train-v2.0.json", return_errors=True)
        dev_data,   dev_errors,   dev_mismatches   = io.read("data/SQuAD/dev-v2.0.json", return_errors=True)
        
        assert len(train_data) == 130_319
        assert len(train_errors) == 0
        assert len(train_mismatches) == 1_009
        assert len(dev_data) == 11873
        assert len(dev_errors) == 0
        assert len(dev_mismatches) == 208
