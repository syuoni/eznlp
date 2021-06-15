# -*- coding: utf-8 -*-
import pytest

from eznlp.utils import ChunksTagsTranslator


@pytest.fixture
def BIOES_tags_example():
    tags = ['O', 'O', 'S-A', 'B-B', 'E-B', 'O', 'S-B', 'O', 'B-C', 'I-C', 'I-C', 'E-C', 'S-C', '<pad>']
    cas_tags = ['O', 'O', 'S', 'B', 'E', 'O', 'S', 'O', 'B', 'I', 'I', 'E', 'S', '<pad>']
    cas_types = ['O', 'O', 'A', 'B', 'B', 'O', 'B', 'O', 'C', 'C', 'C', 'C', 'C', '<pad>']
    
    tag_ids = [1, 1, 5, 6, 8, 1, 9, 1, 10, 11, 11, 12, 13, 0]
    cas_tag_ids = [1, 1, 5, 2, 4, 1, 5, 1, 2, 3, 3, 4, 5, 0]
    cas_type_ids = [1, 1, 2, 3, 3, 1, 3, 1, 4, 4, 4, 4, 4, 0]
    return (tags, cas_tags, cas_types), (tag_ids, cas_tag_ids, cas_type_ids)


def test_tags2chunks(BIOES_tags_example):
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
def test_translation_between_schemes1(from_scheme, from_tags, to_scheme, to_tags):
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
def test_translation_between_schemes2(from_scheme, from_tags, to_scheme, to_tags):
    from_translator = ChunksTagsTranslator(scheme=from_scheme)
    to_translator = ChunksTagsTranslator(scheme=to_scheme)
    
    chunks = from_translator.tags2chunks(from_tags)
    from_tags_translated = to_translator.chunks2tags(chunks, len(from_tags))
    assert from_tags_translated == to_tags
