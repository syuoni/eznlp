# -*- coding: utf-8 -*-
import pytest

from eznlp.token import TokenSequence
from eznlp.relation_classification import RelationClassificationDecoderConfig, RelationClassifierConfig
from eznlp.relation_classification import RelationClassificationDataset


class TestRelationClassificationDataset(object):
    @pytest.mark.parametrize("num_neg_relations, training", [(1, True), 
                                                             (1, False)])
    def test_spans(self, num_neg_relations, training):
        tokenized_raw_text = ["This", "is", "a", "-3.14", "demo", ".", 
                              "Those", "are", "an", "APPLE", "and", "some", "glass", "bottles", "."]
        tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
        
        entities = [{'type': 'EntA', 'start': 4, 'end': 5},
                    {'type': 'EntA', 'start': 9, 'end': 10},
                    {'type': 'EntB', 'start': 12, 'end': 14}]
        chunks = [(ent['type'], ent['start'], ent['end']) for ent in entities]
        
        raw_relations = [{'type': 'RelA', 'head': 0, 'tail': 1}, 
                         {'type': 'RelA', 'head': 0, 'tail': 2}, 
                         {'type': 'RelB', 'head': 1, 'tail': 2}, 
                         {'type': 'RelB', 'head': 2, 'tail': 1}]
        relations = [(rel['type'], chunks[rel['head']], chunks[rel['tail']]) for rel in raw_relations]
        data = [{'tokens': tokens, 'chunks': chunks, 'relations': relations}]
        
        config = RelationClassifierConfig(decoder=RelationClassificationDecoderConfig(num_neg_relations=num_neg_relations))
        dataset = RelationClassificationDataset(data, config, training=training)
        dataset.build_vocabs_and_dims()
        
        span_pairs_obj = dataset[0]['span_pairs_obj']
        assert span_pairs_obj.relations == relations
        assert set((rel[1][1], rel[1][2], rel[2][1], rel[2][2]) for rel in relations).issubset(set(span_pairs_obj.sp_pairs))
        assert (span_pairs_obj.sp_pair_size_ids+1).tolist() == [[he-hs, te-ts] for hs, he, ts, te in span_pairs_obj.sp_pairs]
        assert (span_pairs_obj.rel_label_ids[:len(relations)] != config.decoder.rel_none_idx).all().item()
        assert (span_pairs_obj.rel_label_ids[len(relations):] == config.decoder.rel_none_idx).all().item()
        
        num_chunks = len(chunks)
        expected_num_sp_pairs = num_chunks * (num_chunks-1)
        if training:
            expected_num_sp_pairs = min(expected_num_sp_pairs, len(relations) + num_neg_relations)
            
        assert len(span_pairs_obj.sp_pairs) == expected_num_sp_pairs
        