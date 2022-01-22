# -*- coding: utf-8 -*-
import pytest
import jieba

from eznlp.io import JsonIO, SQuADIO, KarpathyIO, TextClsIO, BratIO
from eznlp.utils.chunk import detect_nested, filter_clashed_by_priority


class TestJsonIO(object):
    """
    
    References
    ----------
    [1] Lu and Roth. 2015. Joint Mention Extraction and Classification with Mention Hypergraphs. EMNLP 2015.
    [2] Eberts and Ulges. 2019. Span-based joint entity and relation extraction with Transformer pre-training. ECAI 2020.
    [3] Luan et al. 2019. A general framework for information extraction using dynamic span graphs. NAACL 2019. 
    [4] Zhong and Chen. 2020. A frustratingly easy approach for joint entity and relation extraction. NAACL 2020. 
    """
    def test_ace2004(self):
        io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        train_data = io.read("data/ace-lu2015emnlp/ACE2004/train.json")
        dev_data   = io.read("data/ace-lu2015emnlp/ACE2004/dev.json")
        test_data  = io.read("data/ace-lu2015emnlp/ACE2004/test.json")
        
        assert len(train_data) == 6_799
        assert sum(len(ex['chunks']) for ex in train_data) == 22_207
        assert max(ck[2]-ck[1] for ex in train_data for ck in ex['chunks']) == 57
        assert len(dev_data) == 829
        assert sum(len(ex['chunks']) for ex in dev_data) == 2_511
        assert max(ck[2]-ck[1] for ex in dev_data for ck in ex['chunks']) == 35
        assert len(test_data) == 879
        assert sum(len(ex['chunks']) for ex in test_data) == 3_031
        assert max(ck[2]-ck[1] for ex in test_data for ck in ex['chunks']) == 43
        
        
    def test_ace2005(self):
        io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        train_data = io.read("data/ace-lu2015emnlp/ACE2005/train.json")
        dev_data   = io.read("data/ace-lu2015emnlp/ACE2005/dev.json")
        test_data  = io.read("data/ace-lu2015emnlp/ACE2005/test.json")
        
        assert len(train_data) == 7_336
        assert sum(len(ex['chunks']) for ex in train_data) == 24_687
        assert max(ck[2]-ck[1] for ex in train_data for ck in ex['chunks']) == 49
        assert len(dev_data) == 958
        assert sum(len(ex['chunks']) for ex in dev_data) == 3_217
        assert max(ck[2]-ck[1] for ex in dev_data for ck in ex['chunks']) == 30
        assert len(test_data) == 1_047
        assert sum(len(ex['chunks']) for ex in test_data) == 3_027
        assert max(ck[2]-ck[1] for ex in test_data for ck in ex['chunks']) == 27
        
        
    def test_genia(self):
        io = JsonIO(text_key='tokens', chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end')
        train_data = io.read("data/genia/term.train.json")
        test_data  = io.read("data/genia/term.test.json")
        
        assert len(train_data) == 16_528
        assert sum(len(ex['chunks']) for ex in train_data) == 50_133
        assert len(test_data) == 1_836
        assert sum(len(ex['chunks']) for ex in test_data) == 5_466
        
        
    def test_ace2004_rel(self):
        io = JsonIO(relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail')
        train_data = io.read("data/ace-luan2019naacl/ace04/cv0.train.json")
        test_data  = io.read("data/ace-luan2019naacl/ace04/cv0.test.json")
        
        # #train + #test = 8683
        assert len(train_data) == 6_898
        assert sum(len(ex['chunks']) for ex in train_data) == 18_065
        assert sum(len(ex['relations']) for ex in train_data) == 3_292
        assert len(test_data) == 1_785
        assert sum(len(ex['chunks']) for ex in test_data) == 4_670
        assert sum(len(ex['relations']) for ex in test_data) == 795
        
        
    def test_ace2005_rel(self):
        io = JsonIO(relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail')
        train_data = io.read("data/ace-luan2019naacl/ace05/train.json")
        dev_data   = io.read("data/ace-luan2019naacl/ace05/dev.json")
        test_data  = io.read("data/ace-luan2019naacl/ace05/test.json")
        
        assert len(train_data) == 10_051
        assert sum(len(ex['chunks']) for ex in train_data) == 26_473
        assert sum(len(ex['relations']) for ex in train_data) == 4_788
        assert len(dev_data) == 2_424
        assert sum(len(ex['chunks']) for ex in dev_data) == 6_338
        assert sum(len(ex['relations']) for ex in dev_data) == 1_131
        assert len(test_data) == 2_050
        assert sum(len(ex['chunks']) for ex in test_data) == 5_476
        assert sum(len(ex['relations']) for ex in test_data) == 1_151
        
        
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
        
        
    def test_yidu_s4k(self):
        io = JsonIO(is_tokenized=False, 
                    tokenize_callback='char', 
                    text_key='originalText', 
                    chunk_key='entities', 
                    chunk_type_key='label_type', 
                    chunk_start_key='start_pos', 
                    chunk_end_key='end_pos', 
                    is_whole_piece=False, 
                    encoding='utf-8-sig')
        train_data1, train_errors1, train_mismatches1 = io.read("data/yidu_s4k/subtask1_training_part1.txt", return_errors=True)
        train_data2, train_errors2, train_mismatches2 = io.read("data/yidu_s4k/subtask1_training_part2.txt", return_errors=True)
        train_data,  train_errors,  train_mismatches  = (train_data1 + train_data2, 
                                                         train_errors1 + train_errors2, 
                                                         train_mismatches1 + train_mismatches2)
        test_data,   test_errors,   test_mismatches   = io.read("data/yidu_s4k/subtask1_test_set_with_answer.json", return_errors=True)
        
        assert len(train_data) == 1_000
        assert sum(len(ex['chunks']) for ex in train_data) == 17_653
        assert len(train_errors) == 0
        assert len(train_mismatches) == 0
        assert len(test_data) == 379
        assert sum(len(ex['chunks']) for ex in test_data) == 6_002
        assert len(test_errors) == 0
        assert len(test_mismatches) == 0
        
        assert not any(detect_nested(ex['chunks']) for data in [train_data, test_data] for ex in data)
        assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=False) == ex['chunks'] for data in [train_data, test_data] for ex in data)
        
        
    def test_cmeee(self):
        io = JsonIO(is_tokenized=False, tokenize_callback='char', text_key='text', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start_idx', chunk_end_key='end_idx', 
                    encoding='utf-8')
        train_data, train_errors, train_mismatches = io.read("data/cblue/CMeEE/CMeEE_train_vz.json", return_errors=True)
        dev_data,   dev_errors,   dev_mismatches   = io.read("data/cblue/CMeEE/CMeEE_dev_vz.json", return_errors=True)
        test_data  = io.read("data/cblue/CMeEE/CMeEE_test_vz.json")
        
        assert len(train_data) == 15_000
        assert sum(len(ex['chunks']) for ex in train_data) == 61_796
        assert len(train_errors) == 0
        assert len(train_mismatches) == 0
        assert len(dev_data) == 5_000
        assert sum(len(ex['chunks']) for ex in dev_data) == 20_300
        assert len(dev_errors) == 0
        assert len(dev_mismatches) == 0
        assert len(test_data) == 3_000
        
        assert any(detect_nested(ex['chunks']) for data in [train_data, dev_data] for ex in data)
        # TODO: clashed chunks?
        # assert all(filter_clashed_by_priority(ex['chunks'], allow_nested=True) == ex['chunks'] for data in [train_data, dev_data] for ex in data)
        
        
    def test_cmeie(self):
        io = JsonIO(is_tokenized=False, tokenize_callback='char', text_key='text', 
                    chunk_key='entities', chunk_type_key='type', chunk_start_key='start', chunk_end_key='end', 
                    relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                    encoding='utf-8')
        train_data = io.read("data/cblue/CMeIE/CMeIE_train_vz.json")
        dev_data   = io.read("data/cblue/CMeIE/CMeIE_dev_vz.json")
        test_data  = io.read("data/cblue/CMeIE/CMeIE_test_vz.json")
        
        assert len(train_data) == 14_339
        assert sum(len(ex['chunks']) for ex in train_data) == 57_880
        assert sum(len(ex['relations']) for ex in train_data) == 43_629
        assert len(dev_data) == 3_585
        assert sum(len(ex['chunks']) for ex in dev_data) == 14_167
        assert sum(len(ex['relations']) for ex in dev_data) == 10_613
        assert len(test_data) == 4_482



@pytest.mark.parametrize("is_whole_piece", [False, True])
def test_read_write_consistency(is_whole_piece):
    brat_io = BratIO(tokenize_callback='char', 
                     has_ins_space=True, ins_space_tokenize_callback=jieba.cut, max_len=200, 
                     parse_attrs=True, parse_relations=True, encoding='utf-8')
    json_io = JsonIO(is_tokenized=True, 
                     attribute_key='attributes', attribute_type_key='type', attribute_chunk_key='entity', 
                     relation_key='relations', relation_type_key='type', relation_head_key='head', relation_tail_key='tail', 
                     is_whole_piece=is_whole_piece, encoding='utf-8')

    src_fn = "data/HwaMei/demo.txt"
    mark = "wp" if is_whole_piece else "nonwp"
    trg_fn = f"data/HwaMei/demo-write-{mark}.json"
    data = brat_io.read(src_fn)
    json_io.write(data, trg_fn)

    data_retr = json_io.read(trg_fn)
    assert data_retr == data



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



class TestKarpathyIO(object):
    """
    References
    ----------
    [1] Karpathy, et al. 2015. Deep visual-semantic alignments for generating image descriptions. CVPR 2015. 
    [2] Vinyals, et al. 2015. Show and tell: A neural image caption generator. CVPR 2015.
    """
    def test_flickr8k(self):
        io = KarpathyIO(img_folder="data/flickr8k/Flicker8k_Dataset")
        train_data, dev_data, test_data = io.read("data/flickr8k/flickr8k-karpathy2015cvpr.json")
        
        assert len(train_data) == 6_000
        assert len(dev_data) == 1_000
        assert len(test_data) == 1_000
        
        
    def test_flickr30k(self):
        io = KarpathyIO(img_folder="data/flickr30k/flickr30k-images")
        train_data, dev_data, test_data = io.read("data/flickr30k/flickr30k-karpathy2015cvpr.json")
        
        assert len(train_data) == 29_000
        assert len(dev_data) == 1_014
        assert len(test_data) == 1_000
        
        
    def test_mscoco(self):
        io = KarpathyIO(img_folder="data/mscoco/data2014")
        train_data, dev_data, test_data = io.read("data/mscoco/mscoco-karpathy2015cvpr.json")
        
        assert len(train_data) == 113_287
        assert len(dev_data) == 5_000
        assert len(test_data) == 5_000



class TestTextClsIO(object):
    """
    References
    ----------
    [1] Zhang et al. 2021. CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark. 
    [2] https://github.com/CBLUEbenchmark/CBLUE
    [3] https://tianchi.aliyun.com/dataset/dataDetail?dataId=113223
    """
    def test_chip_ctc(self):
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='text', mapping={" ": ""}, encoding='utf-8')
        train_data = io.read("data/cblue/CHIP-CTC/CHIP-CTC_train.json")
        dev_data   = io.read("data/cblue/CHIP-CTC/CHIP-CTC_dev.json")
        test_data  = io.read("data/cblue/CHIP-CTC/CHIP-CTC_test.json")
        
        assert len(train_data) == 22_962
        assert len(dev_data) == 7_682
        assert len(test_data) == 10_192  # 10_000?
        
        
    def test_chip_sts(self):
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='text1', paired_text_key='text2', mapping={" ": ""}, encoding='utf-8')
        train_data = io.read("data/cblue/CHIP-STS/CHIP-STS_train.json")
        dev_data   = io.read("data/cblue/CHIP-STS/CHIP-STS_dev.json")
        test_data  = io.read("data/cblue/CHIP-STS/CHIP-STS_test.json")
        
        assert len(train_data) == 16_000
        assert len(dev_data) == 4_000
        assert len(test_data) == 10_000
        
        
    def test_kuake_qic(self):
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query', mapping={" ": ""}, encoding='utf-8')
        train_data = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QIC/KUAKE-QIC_test.json")
        
        assert len(train_data) == 6_931
        assert len(dev_data) == 1_955
        assert len(test_data) == 1_994  # 1_944?
        
        
    def test_kuake_qtr(self):
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query', paired_text_key='title', mapping={" ": ""}, encoding='utf-8')
        train_data = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QTR/KUAKE-QTR_test.json")
        
        assert len(train_data) == 24_174
        assert len(dev_data) == 2_913
        assert len(test_data) == 5_465
        
        
    def test_kuake_qqr(self):
        io = TextClsIO(is_tokenized=False, tokenize_callback=jieba.tokenize, text_key='query1', paired_text_key='query2', mapping={" ": ""}, encoding='utf-8')
        train_data = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_train.json")
        dev_data   = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_dev.json")
        test_data  = io.read("data/cblue/KUAKE-QQR/KUAKE-QQR_test.json")
        
        assert len(train_data) == 15_000
        assert len(dev_data) == 1_599  # 1_600
        assert len(test_data) == 1_596
