# -*- coding: utf-8 -*-
import jieba
import pytest
from eznlp.sequence_tagging.io import ConllIO, BratIO


class TestConllIO(object):
    """
    References
    ----------
    [1] Huang et al. 2015. Bidirectional LSTM-CRF models for sequence tagging. 
    [2] Chiu and Nichols. 2016. Named entity recognition with bidirectional LSTM-CNNs.
    [3] Jie and Lu. 2019. Dependency-guided LSTM-CRF for named entity recognition. 
    [4] Zhang and Yang. 2018. Chinese NER Using Lattice LSTM. 
    [5] Ma et al. 2020. Simplify the Usage of Lexicon in Chinese NER. 
    """
    def test_conll2003(self):
        conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', additional_col_id2name={1: 'pos_tag'})
        train_data = conll_io.read("data/conll2003/eng.train")
        dev_data   = conll_io.read("data/conll2003/eng.testa")
        test_data  = conll_io.read("data/conll2003/eng.testb")
        
        assert len(train_data) == 14_987
        assert sum(len(ex['chunks']) for ex in train_data) == 23_499
        assert sum(len(ex['tokens']) for ex in train_data) == 204_567
        assert len(dev_data) == 3_466
        assert sum(len(ex['chunks']) for ex in dev_data) == 5_942
        assert sum(len(ex['tokens']) for ex in dev_data) == 51_578
        assert len(test_data) == 3_684
        assert sum(len(ex['chunks']) for ex in test_data) == 5_648
        assert sum(len(ex['tokens']) for ex in test_data) == 46_666
        
        assert hasattr(train_data[0]['tokens'][0], 'pos_tag')
        assert train_data[0]['tokens'][0].pos_tag == '-X-'
        
        
    @pytest.mark.slow
    def test_ontonotes5(self):
        conll_io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', line_sep_starts=["#begin", "#end", "pt/"])
        train_data = conll_io.read("data/conll2012/train.english.v4_gold_conll", encoding='utf-8')
        dev_data   = conll_io.read("data/conll2012/dev.english.v4_gold_conll", encoding='utf-8')
        test_data  = conll_io.read("data/conll2012/test.english.v4_gold_conll", encoding='utf-8')
        
        assert len(train_data) == 59_924
        assert sum(len(ex['chunks']) for ex in train_data) == 81_828
        assert sum(len(ex['tokens']) for ex in train_data) == 1_088_503
        assert len(dev_data) == 8_528
        assert sum(len(ex['chunks']) for ex in dev_data) == 11_066
        assert sum(len(ex['tokens']) for ex in dev_data) == 147_724
        assert len(test_data) == 8_262
        assert sum(len(ex['chunks']) for ex in test_data) == 11_257
        assert sum(len(ex['tokens']) for ex in test_data) == 152_728
        
        
    @pytest.mark.slow
    def test_ontonotes5_chinese(self):
        conll_io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', line_sep_starts=["#begin", "#end", "pt/"])
        train_data = conll_io.read("data/conll2012/train.chinese.v4_gold_conll", encoding='utf-8')
        dev_data   = conll_io.read("data/conll2012/dev.chinese.v4_gold_conll", encoding='utf-8')
        test_data  = conll_io.read("data/conll2012/test.chinese.v4_gold_conll", encoding='utf-8')
        
        assert len(train_data) == 36_487
        assert sum(len(ex['chunks']) for ex in train_data) == 62_543
        assert sum(len(ex['tokens']) for ex in train_data) == 756_063
        assert len(dev_data) == 6_083
        assert sum(len(ex['chunks']) for ex in dev_data) == 9_104
        assert sum(len(ex['tokens']) for ex in dev_data) == 110_034
        assert len(test_data) == 4_472
        assert sum(len(ex['chunks']) for ex in test_data) == 7_494
        assert sum(len(ex['tokens']) for ex in test_data) == 92_308
        
        
    def test_sighan2006(self):
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2')
        train_data = conll_io.read("data/SIGHAN2006/train.txt", encoding='utf-8')
        test_data  = conll_io.read("data/SIGHAN2006/test.txt", encoding='utf-8')
        
        assert len(train_data) == 46_364
        assert sum(len(ex['chunks']) for ex in train_data) == 74_703
        assert sum(len(ex['tokens']) for ex in train_data) == 2_169_879
        assert len(test_data) == 4_365
        assert sum(len(ex['chunks']) for ex in test_data) == 6_181
        assert sum(len(ex['tokens']) for ex in test_data) == 172_601
        
        
    def test_resume_ner(self):
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BMES')
        train_data = conll_io.read("data/ResumeNER/train.char.bmes", encoding='utf-8')
        dev_data   = conll_io.read("data/ResumeNER/dev.char.bmes", encoding='utf-8')
        test_data  = conll_io.read("data/ResumeNER/test.char.bmes", encoding='utf-8')
        
        assert len(train_data) == 3_821
        assert sum(len(ex['chunks']) for ex in train_data) == 13_440
        assert sum(len(ex['tokens']) for ex in train_data) == 124_099
        assert len(dev_data) == 463
        assert sum(len(ex['chunks']) for ex in dev_data) == 1_497
        assert sum(len(ex['tokens']) for ex in dev_data) == 13_890
        assert len(test_data) == 477
        assert sum(len(ex['chunks']) for ex in test_data) == 1_630
        assert sum(len(ex['tokens']) for ex in test_data) == 15_100
        
        
    def test_weibo_ner(self):
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2')
        train_data = conll_io.read("data/WeiboNER/weiboNER.conll.train", encoding='utf-8')
        dev_data   = conll_io.read("data/WeiboNER/weiboNER.conll.dev", encoding='utf-8')
        test_data  = conll_io.read("data/WeiboNER/weiboNER.conll.test", encoding='utf-8')
        
        assert len(train_data) == 1_350
        assert sum(len(ex['chunks']) for ex in train_data) == 1_391
        assert sum(len(ex['tokens']) for ex in train_data) == 73_780
        assert len(dev_data) == 270
        assert sum(len(ex['chunks']) for ex in dev_data) == 305
        assert sum(len(ex['tokens']) for ex in dev_data) == 14_509
        assert len(test_data) == 270
        assert sum(len(ex['chunks']) for ex in test_data) == 320
        assert sum(len(ex['tokens']) for ex in test_data) == 14_844
        
        
    def test_weibo_ner_2nd(self):
        conll_io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', pre_text_normalizer=lambda x: x[0])
        train_data = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.train", encoding='utf-8')
        dev_data   = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.dev", encoding='utf-8')
        test_data  = conll_io.read("data/WeiboNER/weiboNER_2nd_conll.test", encoding='utf-8')
        
        assert len(train_data) == 1_350
        assert sum(len(ex['chunks']) for ex in train_data) == 1_895
        assert sum(len(ex['tokens']) for ex in train_data) == 73_778
        assert len(dev_data) == 270
        assert sum(len(ex['chunks']) for ex in dev_data) == 389
        assert sum(len(ex['tokens']) for ex in dev_data) == 14_509
        assert len(test_data) == 270
        assert sum(len(ex['chunks']) for ex in test_data) == 418
        assert sum(len(ex['tokens']) for ex in test_data) == 14_842
        
        
        
class TestBratIO(object):
    @pytest.mark.parametrize("pre_inserted_spaces", [False, True])
    def test_demo(self, pre_inserted_spaces):
        brat_io = BratIO(attr_names=['Denied', 'Analyzed'], 
                         pre_inserted_spaces=pre_inserted_spaces, 
                         tokenize_callback=jieba.cut)
        
        src_fn = "data/brat/demo.txt"
        mark = "spaces" if pre_inserted_spaces else "nospaces"
        trg_fn = f"data/brat/demo-write-{mark}.txt"
        data = brat_io.read(src_fn, encoding='utf-8')
        brat_io.write(data, trg_fn, encoding='utf-8')
        
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
        
        