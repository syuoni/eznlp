# -*- coding: utf-8 -*-
import pytest
import spacy

from eznlp import TokenSequence
from eznlp.sequence_tagging.data_utils import find_ascending, tags2simple_entities
from eznlp.sequence_tagging import entities2tags, tags2entities
from eznlp.sequence_tagging.datasets import TagHelper


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
    return tags, cas_tags, cas_types

@pytest.fixture
def BIOES_tag_ids_example():
    tag_ids = [1, 1, 5, 6, 8, 1, 9, 1, 10, 11, 11, 12, 13, 0]
    cas_tag_ids = [1, 1, 5, 2, 4, 1, 5, 1, 2, 3, 3, 4, 5, 0]
    cas_type_ids = [1, 1, 2, 3, 3, 1, 3, 1, 4, 4, 4, 4, 4, 0]
    return tag_ids, cas_tag_ids, cas_type_ids

@pytest.fixture
def BIOES_tag_helper_example():
    idx2tag = ['<pad>', 'O', 'B-A', 'I-A', 'E-A', 'S-A', 'B-B', 'I-B', 'E-B', 'S-B', 'B-C', 'I-C', 'E-C', 'S-C']
    tag2idx = {t: i for i, t in enumerate(idx2tag)}
    idx2cas_tag = ['<pad>', 'O', 'B', 'I', 'E', 'S']
    cas_tag2idx = {t: i for i, t in enumerate(idx2cas_tag)}
    idx2cas_type = ['<pad>', 'O', 'A', 'B', 'C']
    cas_type2idx = {t: i for i, t in enumerate(idx2cas_type)}
    tag_helper = TagHelper(cascade_mode='None', labeling='BIOES')
    tag_helper.set_vocabs(idx2tag, tag2idx, idx2cas_tag, cas_tag2idx, idx2cas_type, cas_type2idx)
    return tag_helper


class TestTags2Entities(object):
    def test_tags2simple_entities(self, BIOES_tags_example):
        tags, cas_tags, _ = BIOES_tags_example
        simple_entities = tags2simple_entities(tags, labeling='BIOES')
        assert len(simple_entities) == 5
        for ent in simple_entities:
            assert all(tag.split('-')[1] == ent['type'] for tag in tags[ent['start']:ent['stop']])
            
        simple_entities = tags2simple_entities(cas_tags, labeling='BIOES')
        assert len(simple_entities) == 5
        for ent in simple_entities:
            assert ent['type'] == '<pseudo-entity>'
            
            
    def test_tags2entities(self):
        nlp = spacy.load("en_core_web_sm")
        raw_text = "This is a -3.14 demo. Those are an APPLE and some glass bottles."
        tokens = TokenSequence.from_raw_text(raw_text, nlp)
        
        entities = [{'entity': 'demo', 'type': 'Ent', 'start': 16, 'end': 20},
                    {'entity': 'APPLE', 'type': 'Ent', 'start': 35, 'end': 40},
                    {'entity': 'glass bottles', 'type': 'Ent', 'start': 50, 'end': 63}]
        tags = ['O', 'O', 'O', 'O', 'S-Ent', 'O', 'O', 'O', 'O', 'S-Ent', 'O', 'O', 'B-Ent', 'E-Ent', 'O']
        
        tags_built, *_ = entities2tags(raw_text, tokens, entities, labeling='BIOES')
        entities_retr = tags2entities(raw_text, tokens, tags, labeling='BIOES')
        assert tags_built == tags
        assert entities_retr == entities
        
        entities_retr_spans = []
        for span in tokens.spans_within_max_length(10):
            entities_retr_spans.extend(tags2entities(raw_text, tokens[span], tags[span], labeling='BIOES'))
        assert entities_retr_spans == entities
        
        
class TestTagHelper(object):
    def test_dictionary(self, BIOES_tags_example, BIOES_tag_ids_example, BIOES_tag_helper_example):
        tags, cas_tags, cas_types = BIOES_tags_example
        tag_ids, cas_tag_ids, cas_type_ids = BIOES_tag_ids_example
        tag_helper = BIOES_tag_helper_example

        assert tag_helper.tags2ids(tags) == tag_ids
        assert tag_helper.ids2tags(tag_ids) == tags
        assert tag_helper.modeling_tags2ids(tags) == tag_ids
        assert tag_helper.ids2modeling_tags(tag_ids) == tags
        
        tag_helper.set_cascade_mode('Sliced')
        assert tag_helper.cas_tags2ids(cas_tags) == cas_tag_ids
        assert tag_helper.ids2cas_tags(cas_tag_ids) == cas_tags
        assert tag_helper.modeling_tags2ids(cas_tags) == cas_tag_ids
        assert tag_helper.ids2modeling_tags(cas_tag_ids) == cas_tags
        
        assert tag_helper.cas_types2ids(cas_types) == cas_type_ids
        assert tag_helper.ids2cas_types(cas_type_ids) == cas_types
        
    def test_cascade_transform(self, BIOES_tags_example):
        tags, cas_tags, cas_types = BIOES_tags_example
        cas_ent_slices = [slice(2, 3), slice(3, 5), slice(6, 7), slice(8, 12), slice(12, 13)]
        cas_ent_types = ['A', 'B', 'B', 'C', 'C']
        tag_helper = TagHelper(labeling='BIOES')
        
        assert tag_helper.build_cas_tags_by_tags(tags) == cas_tags
        assert tag_helper.build_cas_types_by_tags(tags) == cas_types
        assert tag_helper.build_cas_ent_slices_and_types_by_tags(tags)[0] == cas_ent_slices
        assert tag_helper.build_cas_ent_slices_and_types_by_tags(tags)[1] == cas_ent_types
        assert tag_helper.build_cas_ent_slices_by_cas_tags(cas_tags) == cas_ent_slices
        assert tag_helper.build_tags_by_cas_tags_and_types(cas_tags, cas_types) == tags
        assert tag_helper.build_tags_by_cas_tags_and_ent_slices_and_types(cas_tags, cas_ent_slices, cas_ent_types) == tags
        
        
        