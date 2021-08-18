# -*- coding: utf-8 -*-
from collections import Counter
import pytest
import numpy

from eznlp.io import ConllIO, PostIO


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
    def _assert_flatten_consistency(self, data):
        flattened_data = self.io.flatten_to_characters(data)
        
        for entry, f_entry in zip(data, flattened_data):
            assert "".join(f_entry['tokens'].raw_text) == "".join(entry['tokens'].raw_text)
            
            chunk_texts = ["".join(entry['tokens'].raw_text[s:e]) for _, s, e in entry['chunks']]
            f_chunk_texts = ["".join(f_entry['tokens'].raw_text[s:e]) for _, s, e in f_entry['chunks']]
            assert f_chunk_texts == chunk_texts
            
            if hasattr(entry['tokens'][0], 'pos_tag'):
                char_seq_lens = [len(tok) for tok in entry['tokens'].raw_text]
                cum_char_seq_lens = [0] + numpy.cumsum(char_seq_lens).tolist()
                assert [f_entry['tokens'].pos_tag[i] for i in cum_char_seq_lens[:-1]] == entry['tokens'].pos_tag
        
        
        
    def test_conll2003(self):
        self.io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', additional_col_id2name={1: 'pos_tag'})
        train_data = self.io.read("data/conll2003/eng.train")
        dev_data   = self.io.read("data/conll2003/eng.testa")
        test_data  = self.io.read("data/conll2003/eng.testb")
        
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
        
        self._assert_flatten_consistency(test_data)
        

    def test_conll2003_at_doc_level(self):
        self.io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1', document_sep_starts=["-DOCSTART-"], document_level=True)
        train_data = self.io.read("data/conll2003/eng.train")
        dev_data   = self.io.read("data/conll2003/eng.testa")
        test_data  = self.io.read("data/conll2003/eng.testb")
        
        assert len(train_data) == 946
        assert sum(len(ex['chunks']) for ex in train_data) == 23_499
        assert sum(len(ex['tokens']) for ex in train_data) == 204_567 - 946
        assert len(dev_data) == 216
        assert sum(len(ex['chunks']) for ex in dev_data) == 5_942
        assert sum(len(ex['tokens']) for ex in dev_data) == 51_578 - 216
        assert len(test_data) == 231
        assert sum(len(ex['chunks']) for ex in test_data) == 5_648
        assert sum(len(ex['tokens']) for ex in test_data) == 46_666 - 231
        
        self._assert_flatten_consistency(test_data)
        

    @pytest.mark.slow
    def test_ontonotesv5(self):
        self.io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', sentence_sep_starts=["#end", "pt/"], document_sep_starts=["#begin"], encoding='utf-8')
        train_data = self.io.read("data/conll2012/train.english.v4_gold_conll")
        dev_data   = self.io.read("data/conll2012/dev.english.v4_gold_conll")
        test_data  = self.io.read("data/conll2012/test.english.v4_gold_conll")
        
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
    def test_ontonotesv5_at_doc_level(self):
        self.io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', sentence_sep_starts=["#end", "pt/"], document_sep_starts=["#begin"], document_level=True, encoding='utf-8')
        train_data = self.io.read("data/conll2012/train.english.v4_gold_conll")
        dev_data   = self.io.read("data/conll2012/dev.english.v4_gold_conll")
        test_data  = self.io.read("data/conll2012/test.english.v4_gold_conll")
        
        assert len(train_data) == 2_483
        assert sum(len(ex['chunks']) for ex in train_data) == 81_828
        assert sum(len(ex['tokens']) for ex in train_data) == 1_088_503
        assert len(dev_data) == 319
        assert sum(len(ex['chunks']) for ex in dev_data) == 11_066
        assert sum(len(ex['tokens']) for ex in dev_data) == 147_724
        assert len(test_data) == 322
        assert sum(len(ex['chunks']) for ex in test_data) == 11_257
        assert sum(len(ex['tokens']) for ex in test_data) == 152_728
        

    @pytest.mark.slow
    def test_ontonotesv5_chinese(self):
        self.io = ConllIO(text_col_id=3, tag_col_id=10, scheme='OntoNotes', sentence_sep_starts=["#end"], document_sep_starts=["#begin"], encoding='utf-8')
        train_data = self.io.read("data/conll2012/train.chinese.v4_gold_conll")
        dev_data   = self.io.read("data/conll2012/dev.chinese.v4_gold_conll")
        test_data  = self.io.read("data/conll2012/test.chinese.v4_gold_conll")
        
        assert len(train_data) == 36_487
        assert sum(len(ex['chunks']) for ex in train_data) == 62_543
        assert sum(len(ex['tokens']) for ex in train_data) == 756_063
        assert len(dev_data) == 6_083
        assert sum(len(ex['chunks']) for ex in dev_data) == 9_104
        assert sum(len(ex['tokens']) for ex in dev_data) == 110_034
        assert len(test_data) == 4_472
        assert sum(len(ex['chunks']) for ex in test_data) == 7_494
        assert sum(len(ex['tokens']) for ex in test_data) == 92_308
        
        self._assert_flatten_consistency(test_data)
        

    @pytest.mark.slow
    def test_ontonotesv4_chinese(self):
        self.io = ConllIO(text_col_id=2, tag_col_id=3, scheme='OntoNotes', sentence_sep_starts=["#end"], document_sep_starts=["#begin"], encoding='utf-8')
        train_data = self.io.read("data/ontonotesv4/train.chinese.vz_gold_conll")
        dev_data   = self.io.read("data/ontonotesv4/dev.chinese.vz_gold_conll")
        test_data  = self.io.read("data/ontonotesv4/test.chinese.vz_gold_conll")
        
        assert len(train_data) == 15_724
        assert sum(len(ex['chunks']) for ex in train_data) == 24_166
        assert sum(len(ex['tokens']) for ex in train_data) == 313_648
        assert len(dev_data) == 4_442
        assert sum(len(ex['chunks']) for ex in dev_data) == 12_287
        assert sum(len(ex['tokens']) for ex in dev_data) == 124_116
        assert len(test_data) == 4_487
        assert sum(len(ex['chunks']) for ex in test_data) == 12_967
        assert sum(len(ex['tokens']) for ex in test_data) == 128_277
        
        self._assert_flatten_consistency(test_data)

        
    @pytest.mark.slow
    def test_sighan2006(self):
        self.io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8')
        train_data = self.io.read("data/SIGHAN2006/train.txt")
        test_data  = self.io.read("data/SIGHAN2006/test.txt")
        
        assert len(train_data) == 46_364
        assert sum(len(ex['chunks']) for ex in train_data) == 74_703
        assert sum(len(ex['tokens']) for ex in train_data) == 2_169_879
        assert len(test_data) == 4_365
        assert sum(len(ex['chunks']) for ex in test_data) == 6_181
        assert sum(len(ex['tokens']) for ex in test_data) == 172_601
        
        
    def test_resume_ner(self):
        self.io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BMES', encoding='utf-8')
        train_data = self.io.read("data/ResumeNER/train.char.bmes")
        dev_data   = self.io.read("data/ResumeNER/dev.char.bmes")
        test_data  = self.io.read("data/ResumeNER/test.char.bmes")
        
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
        self.io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8')
        train_data = self.io.read("data/WeiboNER/weiboNER.conll.train")
        dev_data   = self.io.read("data/WeiboNER/weiboNER.conll.dev")
        test_data  = self.io.read("data/WeiboNER/weiboNER.conll.test")
        
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
        self.io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', encoding='utf-8', pre_text_normalizer=lambda x: x[0])
        train_data = self.io.read("data/WeiboNER/weiboNER_2nd_conll.train")
        dev_data   = self.io.read("data/WeiboNER/weiboNER_2nd_conll.dev")
        test_data  = self.io.read("data/WeiboNER/weiboNER_2nd_conll.test")
        
        assert len(train_data) == 1_350
        assert sum(len(ex['chunks']) for ex in train_data) == 1_895
        assert sum(len(ex['tokens']) for ex in train_data) == 73_778
        assert len(dev_data) == 270
        assert sum(len(ex['chunks']) for ex in dev_data) == 389
        assert sum(len(ex['tokens']) for ex in dev_data) == 14_509
        assert len(test_data) == 270
        assert sum(len(ex['chunks']) for ex in test_data) == 418
        assert sum(len(ex['tokens']) for ex in test_data) == 14_842
        
        
    def test_clerd(self):
        self.io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BIO2', tag_sep='_', encoding='utf-8')
        train_data = self.io.read("data/CLERD/ner/train.txt")
        dev_data   = self.io.read("data/CLERD/ner/validation.txt")
        test_data  = self.io.read("data/CLERD/ner/test.txt")
        
        assert len(train_data) == 24_165
        assert sum(len(ex['chunks']) for ex in train_data) == 133_105
        assert len(dev_data) == 1_895
        assert sum(len(ex['chunks']) for ex in dev_data) == 10_571
        assert len(test_data) == 2_837
        assert sum(len(ex['chunks']) for ex in test_data) == 16_186
        
        # Check post-IO processing
        data = train_data + dev_data + test_data
        ck_counter = Counter(ck[0] for entry in data for ck in entry['chunks'])
        assert max(ck[2]-ck[1] for entry in data for ck in entry['chunks']) == 315
        
        post_io = PostIO(verbose=False)
        data = post_io.map(data, 
                           chunk_type_mapping=lambda x: x.title() if x not in ('Physical', 'Term') else None, max_span_size=20, 
                           relation_type_mapping=lambda x: x if x not in ('Coreference', ) else None)
        
        post_ck_counter = Counter(ck[0] for entry in data for ck in entry['chunks'])
        assert max(ck[2]-ck[1] for entry in data for ck in entry['chunks']) == 20
        assert len(post_ck_counter) == 7
        assert sum(ck_counter.values()) - sum(post_ck_counter.values()) == 154
