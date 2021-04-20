# -*- coding: utf-8 -*-
import jieba
import pytest

from eznlp.io import BratIO


class TestBratIO(object):
    @pytest.mark.parametrize("pre_inserted_spaces", [False, True])
    def test_demo(self, pre_inserted_spaces):
        brat_io = BratIO(attr_names=['Denied', 'Analyzed'], 
                         pre_inserted_spaces=pre_inserted_spaces, 
                         tokenize_callback=jieba.cut, 
                         encoding='utf-8')
        
        src_fn = "data/brat/demo.txt"
        mark = "spaces" if pre_inserted_spaces else "nospaces"
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
        
        