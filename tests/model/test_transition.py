# -*- coding: utf-8 -*-
import pytest

from eznlp.token import TokenSequence
from eznlp.model.decoder.transition import ChunksTagsTranslator


@pytest.fixture
def BIOES_tags_example():
    tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
    cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
    cas_types = ['O', 'O', 'A', 'B', 'B', 'O', 'B', 'O', 'C', 'C', 'C', 'C', 'C', '<pad>']
    
    tag_ids = [1, 1, 5, 6, 8, 1, 9, 1, 10, 11, 11, 12, 13, 0]
    cas_tag_ids = [1, 1, 5, 2, 4, 1, 5, 1, 2, 3, 3, 4, 5, 0]
    cas_type_ids = [1, 1, 2, 3, 3, 1, 3, 1, 4, 4, 4, 4, 4, 0]
    return (tags, cas_tags, cas_types), (tag_ids, cas_tag_ids, cas_type_ids)


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
            
            
    def test_tags2text_chunks(self, spacy_nlp_en):
        raw_text = "This is a -3.14 demo. Those are an APPLE and some glass bottles."
        tokens = TokenSequence.from_raw_text(raw_text, spacy_nlp_en)
        
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
        
    
    @pytest.mark.parametrize("from_scheme, from_tags", 
                             [('BIO1',  ['O', 'I-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'I-EntC', 'I-EntC', 'O']), 
                              ('BIO2',  ['O', 'B-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'B-EntC', 'I-EntC', 'O']), 
                              ('BIOES', ['O', 'B-EntA', 'I-EntA', 'E-EntA', 'S-EntA', 'S-EntB', 'O', 'B-EntC', 'E-EntC', 'O']), 
                              ('OntoNotes', ['*', '(EntA*', '*', '*)', '(EntA)', '(EntB)', '*', '(EntC*', '*)', '*'])])
    @pytest.mark.parametrize("to_scheme, to_tags", 
                             [('BIO1',  ['O', 'I-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'I-EntC', 'I-EntC', 'O']), 
                              ('BIO2',  ['O', 'B-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'B-EntC', 'I-EntC', 'O']), 
                              ('BIOES', ['O', 'B-EntA', 'I-EntA', 'E-EntA', 'S-EntA', 'S-EntB', 'O', 'B-EntC', 'E-EntC', 'O']), 
                              ('OntoNotes', ['*', '(EntA*', '*', '*)', '(EntA)', '(EntB)', '*', '(EntC*', '*)', '*'])])
    def test_scheme_translation1(self, from_scheme, from_tags, to_scheme, to_tags):
        from_translator = ChunksTagsTranslator(scheme=from_scheme)
        to_translator = ChunksTagsTranslator(scheme=to_scheme)
        
        chunks = from_translator.tags2chunks(from_tags)
        from_tags_translated = to_translator.chunks2tags(chunks, len(from_tags))
        assert from_tags_translated == to_tags
        
    @pytest.mark.parametrize("from_scheme, from_tags", 
                             [('BIO1',  ['I-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'I-EntC', 'I-EntC']), 
                              ('BIO2',  ['B-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'B-EntC', 'I-EntC']), 
                              ('BIOES', ['B-EntA', 'I-EntA', 'E-EntA', 'S-EntA', 'S-EntB', 'O', 'B-EntC', 'E-EntC']), 
                              ('OntoNotes', ['(EntA*', '*', '*)', '(EntA)', '(EntB)', '*', '(EntC*', '*)'])])
    @pytest.mark.parametrize("to_scheme, to_tags", 
                             [('BIO1',  ['I-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'I-EntC', 'I-EntC']), 
                              ('BIO2',  ['B-EntA', 'I-EntA', 'I-EntA', 'B-EntA', 'B-EntB', 'O', 'B-EntC', 'I-EntC']), 
                              ('BIOES', ['B-EntA', 'I-EntA', 'E-EntA', 'S-EntA', 'S-EntB', 'O', 'B-EntC', 'E-EntC']), 
                              ('OntoNotes', ['(EntA*', '*', '*)', '(EntA)', '(EntB)', '*', '(EntC*', '*)'])])
    def test_scheme_translation2(self, from_scheme, from_tags, to_scheme, to_tags):
        from_translator = ChunksTagsTranslator(scheme=from_scheme)
        to_translator = ChunksTagsTranslator(scheme=to_scheme)
        
        chunks = from_translator.tags2chunks(from_tags)
        from_tags_translated = to_translator.chunks2tags(chunks, len(from_tags))
        assert from_tags_translated == to_tags
        