# -*- coding: utf-8 -*-
import pytest
import spacy
import numpy as np

from eznlp import TokenSequence
from eznlp.sequence_tagging import DecoderConfig
from eznlp.sequence_tagging import ChunksTagsTranslator, SchemeTranslator
from eznlp.sequence_tagging import precision_recall_f1_report
from eznlp.sequence_tagging.raw_data import parse_conll_file
from eznlp.sequence_tagging.transition import find_ascending


def test_find_ascending():
    for v in [-3, 2, 2.5, 9]:
        x = list(range(5))
        find, idx = find_ascending(x, v)
        x.insert(idx, v)
        
        assert find == (v in list(range(5)))
        assert len(x) == 6
        assert all(x[i] <= x[i+1] for i in range(len(x)-1))


@pytest.fixture
def BIOES_tags_example():
    tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
    cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
    cas_types = ['O', 'O', 'A', 'B', 'B', 'O', 'B', 'O', 'C', 'C', 'C', 'C', 'C', '<pad>']
    
    tag_ids = [1, 1, 5, 6, 8, 1, 9, 1, 10, 11, 11, 12, 13, 0]
    cas_tag_ids = [1, 1, 5, 2, 4, 1, 5, 1, 2, 3, 3, 4, 5, 0]
    cas_type_ids = [1, 1, 2, 3, 3, 1, 3, 1, 4, 4, 4, 4, 4, 0]
    return (tags, cas_tags, cas_types), (tag_ids, cas_tag_ids, cas_type_ids)


@pytest.fixture
def BIOES_dec_config_example():
    idx2tag = ['<pad>', 'O', 'B-A', 'I-A', 'E-A', 'S-A', 'B-B', 'I-B', 'E-B', 'S-B', 'B-C', 'I-C', 'E-C', 'S-C']
    idx2cas_tag = ['<pad>', 'O', 'B', 'I', 'E', 'S']
    idx2cas_type = ['<pad>', 'O', 'A', 'B', 'C']
    
    dec_config_nocas = DecoderConfig(scheme='BIOES', cascade_mode='None', 
                                     idx2tag=idx2tag, idx2cas_tag=idx2cas_tag, idx2cas_type=idx2cas_type)
    dec_config_sliced = DecoderConfig(scheme='BIOES', cascade_mode='Sliced', 
                                   idx2tag=idx2tag, idx2cas_tag=idx2cas_tag, idx2cas_type=idx2cas_type)
    return dec_config_nocas, dec_config_sliced


class TestChunksTagsTranslator(object):
    def test_tags2chunks(self, BIOES_tags_example):
        (tags, cas_tags, _), *_ = BIOES_tags_example
        
        translator = ChunksTagsTranslator(scheme='BIOES')
        chunks = translator.tags2chunks(tags)
        assert len(chunks) == 5
        for chunk_type, chunk_start, chunk_end in chunks:
            assert all(tag.split('-')[1] == chunk_type for tag in tags[chunk_start:chunk_end])
            
        chunks = translator.tags2chunks(cas_tags)
        assert len(chunks) == 5
        for chunk_type, chunk_start, chunk_end in chunks:
            assert chunk_type == '<pseudo-type>'
            
            
    def test_tags2text_chunks(self):
        nlp = spacy.load("en_core_web_sm")
        raw_text = "This is a -3.14 demo. Those are an APPLE and some glass bottles."
        tokens = TokenSequence.from_raw_text(raw_text, nlp)
        
        entities = [{'entity': 'demo', 'type': 'EntA', 'start': 16, 'end': 20},
                    {'entity': 'APPLE', 'type': 'EntA', 'start': 35, 'end': 40},
                    {'entity': 'glass bottles', 'type': 'EntB', 'start': 50, 'end': 63}]
        text_chunks = [(ent['entity'], ent['type'], ent['start'], ent['end']) for ent in entities]
        tags = ['O', 'O', 'O', 'O', 'S-EntA', 'O', 'O', 'O', 'O', 'S-EntA', 'O', 'O', 'B-EntB', 'E-EntC', 'O']
        
        translator = ChunksTagsTranslator(scheme='BIOES')
        tags_built, *_ = translator.text_chunks2tags(text_chunks, raw_text, tokens)
        text_chunks_retr = translator.tags2text_chunks(tags, raw_text, tokens, breaking_for_types=False)
        
        assert tags_built[-2] == 'E-EntB'
        tags_built[-2] = 'E-EntC'
        assert tags_built == tags
        assert text_chunks_retr == text_chunks
        
        text_chunks_retr_spans = []
        for span in tokens.spans_within_max_length(10):
            text_chunks_retr_spans.extend(translator.tags2text_chunks(tags[span], raw_text, tokens[span], breaking_for_types=False))
        assert text_chunks_retr_spans == text_chunks
        
        
        text_chunks_retr = translator.tags2text_chunks(tags, raw_text, tokens, breaking_for_types=True)
        assert text_chunks_retr[:2] == text_chunks[:2]
        assert text_chunks_retr[2][1] == 'EntB'
        assert text_chunks_retr[3][1] == 'EntC'
        
        
class TestSchemeTranslator(object):
    def test_scheme_translator(self):
        tags_dic = {'BIO1':  ['O', 'I-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'I-EntC', 'I-EntC', 'O'], 
                    'BIO2':  ['O', 'B-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'B-EntC', 'I-EntC', 'O'], 
                    'BIOES': ['O', 'B-EntA', 'I-EntA', 'E-EntA', 'S-EntA', 'S-EntB', 'O', 'B-EntC', 'E-EntC', 'O']}
        
        for from_scheme in tags_dic:
            for to_scheme in tags_dic:
                from_tags_translated = SchemeTranslator(from_scheme=from_scheme, to_scheme=to_scheme, 
                                                        breaking_for_types=True).translate(tags_dic[from_scheme])
                assert from_tags_translated == tags_dic[to_scheme]
                
                
class TestTagHelper(object):
    def test_dictionary(self, BIOES_tags_example, BIOES_dec_config_example):
        (tags, cas_tags, cas_types), (tag_ids, cas_tag_ids, cas_type_ids) = BIOES_tags_example
        dec_config, _ = BIOES_dec_config_example

        assert dec_config.tags2ids(tags) == tag_ids
        assert dec_config.ids2tags(tag_ids) == tags
        assert dec_config.modeling_tags2ids(tags) == tag_ids
        assert dec_config.ids2modeling_tags(tag_ids) == tags
        
        _, dec_config = BIOES_dec_config_example
        assert dec_config.cas_tags2ids(cas_tags) == cas_tag_ids
        assert dec_config.ids2cas_tags(cas_tag_ids) == cas_tags
        assert dec_config.modeling_tags2ids(cas_tags) == cas_tag_ids
        assert dec_config.ids2modeling_tags(cas_tag_ids) == cas_tags
        
        assert dec_config.cas_types2ids(cas_types) == cas_type_ids
        assert dec_config.ids2cas_types(cas_type_ids) == cas_types
        
        
    def test_cascade_transform(self, BIOES_tags_example, BIOES_dec_config_example):
        (tags, cas_tags, cas_types), *_ = BIOES_tags_example
        dec_config, _ = BIOES_dec_config_example
        cas_ent_slices = [slice(2, 3), slice(3, 5), slice(6, 7), slice(8, 12), slice(12, 13)]
        cas_ent_types = ['A', 'B', 'B', 'C', 'C']
        
        assert dec_config.build_cas_tags_by_tags(tags) == cas_tags
        assert dec_config.build_cas_types_by_tags(tags) == cas_types
        assert dec_config.build_cas_ent_slices_and_types_by_tags(tags)[0] == cas_ent_slices
        assert dec_config.build_cas_ent_slices_and_types_by_tags(tags)[1] == cas_ent_types
        assert dec_config.build_cas_ent_slices_by_cas_tags(cas_tags) == cas_ent_slices
        assert dec_config.build_tags_by_cas_tags_and_types(cas_tags, cas_types) == tags
        assert dec_config.build_tags_by_cas_tags_and_ent_slices_and_types(cas_tags, cas_ent_slices, cas_ent_types) == tags
        
        
class TestMetrics(object):
    def one_pair_pass(self, tags_gold_data, tags_pred_data, expected_ave_scores):
        translator = ChunksTagsTranslator(scheme='BIO2')
        chunks_gold_data = [translator.tags2chunks(tags, False) for tags in tags_gold_data]
        chunks_pred_data = [translator.tags2chunks(tags, False) for tags in tags_pred_data]
        
        scores, ave_scores = precision_recall_f1_report(chunks_gold_data, chunks_pred_data)
        
        for key in ave_scores:
            for score_key in ['precision', 'recall', 'f1']:
                assert np.abs(ave_scores[key][score_key] - expected_ave_scores[key][score_key]) < 1e-6
                
        
    def test_example1(self):
        tags_gold_data = [['B-A', 'B-B', 'O', 'B-A']]
        tags_pred_data = [['O', 'B-B', 'B-C', 'B-A']]
        expected_ave_scores = {'macro': {'precision': 2/3,
                                         'recall': 1/2, 
                                         'f1': 5/9}, 
                               'micro': {'precision': 2/3, 
                                         'recall': 2/3, 
                                         'f1': 2/3}}
        self.one_pair_pass(tags_gold_data, tags_pred_data, expected_ave_scores)
        
        
    def test_example2(self):
        tags_gold_data = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        tags_pred_data = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        expected_ave_scores = {'macro': {'precision': 0.5,
                                         'recall': 0.5, 
                                         'f1': 0.5}, 
                               'micro': {'precision': 0.5,
                                         'recall': 0.5, 
                                         'f1': 0.5}}
        self.one_pair_pass(tags_gold_data, tags_pred_data, expected_ave_scores)
        
        
    def test_conll2000(self):
        # https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        # https://github.com/chakki-works/seqeval
        conll_config = {'raw_scheme': 'BIO1', 
                        'scheme': 'BIO2', 
                        'columns': ['text', 'pos_tag', 'chunking_tag_gold', 'chunking_tag_pred'], 
                        'attach_additional_tags': False, 
                        'skip_docstart': False, 
                        'lower_case_mode': 'None'}
        
        gold_data = parse_conll_file("assets/conlleval/output.txt", trg_col='chunking_tag_gold', **conll_config)
        pred_data = parse_conll_file("assets/conlleval/output.txt", trg_col='chunking_tag_pred', **conll_config)
        
        tags_gold_data = [ex['tags'] for ex in gold_data]
        tags_pred_data = [ex['tags'] for ex in pred_data]
        
        expected_ave_scores = {'macro': {'precision': 0.54880502,
                                         'recall': 0.58776420, 
                                         'f1': 0.55397552}, 
                               'micro': {'precision': 0.68831169, 
                                         'recall': 0.80827887, 
                                         'f1': 0.74348697}}
        self.one_pair_pass(tags_gold_data, tags_pred_data, expected_ave_scores)
        
        