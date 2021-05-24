# -*- coding: utf-8 -*-
import jieba
import pytest

from eznlp.io import BratIO


class TestBratIO(object):
    @pytest.mark.parametrize("has_ins_space", [False, True])
    def test_demo(self, has_ins_space):
        brat_io = BratIO(attr_names=['Denied', 'Analyzed'], 
                         has_ins_space=has_ins_space, 
                         ins_space_tokenize_callback=jieba.cut if has_ins_space else None, 
                         encoding='utf-8')
        
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
        
        