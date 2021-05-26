# -*- coding: utf-8 -*-
import jieba
import pytest

from eznlp.io import BratIO


class TestBratIO(object):
    """
    
    References
    ----------
    [1] Xu et al. 2017. A discourse-level named entity recognition and relation extraction dataset for chinese literature text. 
    """
    def test_clerd(self):
        self.io = BratIO(tokenize_callback='char', has_ins_space=False, parse_attrs=False, parse_relations=True, 
                         max_len=500, line_sep="\n", allow_broken_chunk_text=True, consistency_mapping={'[・;é]': '、'}, encoding='utf-8')
        train_data = self.io.read_folder("data/CLERD/relation_extraction/Training")
        dev_data   = self.io.read_folder("data/CLERD/relation_extraction/Validation")
        test_data  = self.io.read_folder("data/CLERD/relation_extraction/Testing")
        
        assert len(train_data) == 2_820
        assert sum(len(ex['chunks']) for ex in train_data) == 124_712
        assert sum(len(ex['relations']) for ex in train_data) == 13_135
        assert len(dev_data) == 264
        assert sum(len(ex['chunks']) for ex in dev_data) == 11_899
        assert sum(len(ex['relations']) for ex in dev_data) == 1_319
        assert len(test_data) == 367
        assert sum(len(ex['chunks']) for ex in test_data) == 16_780
        assert sum(len(ex['relations']) for ex in test_data) == 1_636




@pytest.mark.parametrize("has_ins_space", [False, True])
def test_read_write_consistency(has_ins_space):
    brat_io = BratIO(tokenize_callback='char', 
                     has_ins_space=has_ins_space, ins_space_tokenize_callback=jieba.cut if has_ins_space else None, 
                     parse_attrs=True, parse_relations=True, encoding='utf-8')
    
    src_fn = "data/brat/demo.txt"
    mark = "spaces" if has_ins_space else "nospaces"
    trg_fn = f"data/brat/demo-write-{mark}.txt"
    data = brat_io.read(src_fn)
    brat_io.write(data, trg_fn)
    
    with open(src_fn, encoding='utf-8') as f:
        gold_text_lines = [line.strip().replace(" ", "") for line in f if line.strip() != ""]
    with open(trg_fn, encoding='utf-8') as f:
        retr_text_lines = [line.strip().replace(" ", "") for line in f if line.strip() != ""]
    assert retr_text_lines == gold_text_lines
    
    with open(src_fn, encoding='utf-8') as f:
        gold_ann_lines = [line.strip().replace(" ", "") for line in f if line.strip() != ""]
    with open(trg_fn, encoding='utf-8') as f:
        retr_ann_lines = [line.strip().replace(" ", "") for line in f if line.strip() != ""]
    assert len(retr_ann_lines) == len(gold_ann_lines)
    
    gold_chunk_anns = [line.split("\t", 1)[1] for line in gold_ann_lines if line.startswith('T')]
    retr_chunk_anns = [line.split("\t", 1)[1] for line in retr_ann_lines if line.startswith('T')]
    assert sorted(retr_chunk_anns) == sorted(gold_chunk_anns)
