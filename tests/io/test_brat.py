# -*- coding: utf-8 -*-
from collections import Counter
import jieba
import pytest

from eznlp.io import BratIO, PostIO


class TestBratIO(object):
    """
    
    References
    ----------
    [1] Xu et al. 2017. A discourse-level named entity recognition and relation extraction dataset for chinese literature text. 
    """
    def test_clerd(self):
        io = BratIO(tokenize_callback='char', has_ins_space=False, parse_attrs=False, parse_relations=True, 
                    max_len=500, line_sep="\n", allow_broken_chunk_text=True, consistency_mapping={'[・;é]': '、'}, encoding='utf-8')
        train_data, train_errors, train_mismatches = io.read_folder("data/CLERD/relation_extraction/Training", return_errors=True)
        dev_data,   dev_errors,   dev_mismatches   = io.read_folder("data/CLERD/relation_extraction/Validation", return_errors=True)
        test_data,  test_errors,  test_mismatches  = io.read_folder("data/CLERD/relation_extraction/Testing", return_errors=True)
        
        assert len(train_data) == 2_820
        assert sum(len(ex['chunks']) for ex in train_data) == 124_712
        assert sum(len(ex['relations']) for ex in train_data) == 13_135
        assert len(train_errors) == 2
        assert len(train_mismatches) == 0
        assert len(dev_data) == 264
        assert sum(len(ex['chunks']) for ex in dev_data) == 11_899
        assert sum(len(ex['relations']) for ex in dev_data) == 1_319
        assert len(dev_errors) == 0
        assert len(dev_mismatches) == 0
        assert len(test_data) == 367
        assert sum(len(ex['chunks']) for ex in test_data) == 16_780
        assert sum(len(ex['relations']) for ex in test_data) == 1_636
        assert len(test_errors) == 0
        assert len(test_mismatches) == 0
        
        # Check post-IO processing
        data = train_data + dev_data + test_data
        ck_counter = Counter(ck[0] for entry in data for ck in entry['chunks'])
        rel_counter = Counter(rel[0] for entry in data for rel in entry['relations'])
        rel_ck_counter = Counter(ck[0] for entry in data for rel in entry['relations'] for ck in rel[1:])
        assert max(ck[2]-ck[1] for entry in data for ck in entry['chunks']) == 205
        
        post_io = PostIO(verbose=False)
        data = post_io.map(data, 
                           chunk_type_mapping=lambda x: x.split('-')[0] if x not in ('Physical', 'Term') else None, max_span_size=20, 
                           relation_type_mapping=lambda x: x if x not in ('Coreference', ) else None)
        
        post_ck_counter = Counter(ck[0] for entry in data for ck in entry['chunks'])
        post_rel_counter = Counter(rel[0] for entry in data for rel in entry['relations'])
        post_rel_ck_counter = Counter(ck[0] for entry in data for rel in entry['relations'] for ck in rel[1:])
        assert max(ck[2]-ck[1] for entry in data for ck in entry['chunks']) == 20
        assert len(post_ck_counter) == 7
        assert sum(ck_counter.values()) - sum(post_ck_counter.values()) == 398
        assert len(post_rel_counter) == 9
        assert sum(rel_counter.values()) - sum(post_rel_counter.values()) == 91
        assert len(post_rel_ck_counter) == 4
        assert sum(rel_ck_counter.values()) - sum(post_rel_ck_counter.values()) == 182




@pytest.mark.parametrize("has_ins_space", [False, True])
def test_read_write_consistency(has_ins_space):
    brat_io = BratIO(tokenize_callback='char', 
                     has_ins_space=has_ins_space, ins_space_tokenize_callback=jieba.cut if has_ins_space else None, 
                     parse_attrs=True, parse_relations=True, encoding='utf-8')
    
    src_fn = "data/HwaMei/demo.txt"
    mark = "spaces" if has_ins_space else "nospaces"
    trg_fn = f"data/HwaMei/demo-write-{mark}.txt"
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
