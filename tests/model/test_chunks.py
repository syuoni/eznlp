# -*- coding: utf-8 -*-
import pytest
import itertools

from eznlp.model import SpanRelClassificationDecoderConfig


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("pipeline", [True, False])
def test_chunk_pairs_obj(training, pipeline, EAR_data_demo):
    if pipeline:
        for entry in EAR_data_demo:
            entry['chunks_pred'] = []
    
    entry = EAR_data_demo[0]
    tokens, chunks, relations = entry['tokens'], entry['chunks'], entry['relations']
    
    config = SpanRelClassificationDecoderConfig(max_span_size=3)
    config.build_vocab(EAR_data_demo)
    cp_obj = config.exemplify(entry, training=training)['cp_obj']
    
    num_tokens = len(tokens)
    num_chunks = len(chunks)
    assert cp_obj.relations == relations
    
    if pipeline and training:
        assert cp_obj.chunks == chunks
        assert cp_obj.cp2label_id.size() == (num_chunks, num_chunks)
        assert cp_obj.cp2label_id.sum() == sum(config.label2idx[label] for label, *_ in relations)
        assert all(cp_obj.cp2label_id[cp_obj.chunks.index(head), cp_obj.chunks.index(tail)] == config.label2idx[label] 
                       for label, head, tail in relations)
        
        labels_retr = [config.idx2label[i] for i in cp_obj.cp2label_id.flatten().tolist()]
        relations_retr = [(label, head, tail) for label, (head, tail) in zip(labels_retr, itertools.product(cp_obj.chunks, cp_obj.chunks)) if label != config.none_label]
        assert set(relations_retr) == set(relations)
        
    elif pipeline and not training:
        assert len(cp_obj.chunks) == 0
        assert cp_obj.cp2label_id.size() == (0, 0)
        
    else:
        assert not hasattr(cp_obj, 'chunks')
        assert not hasattr(cp_obj, 'cp2label_id')
        
        chunks_pred = [('EntA', 0, 1), ('EntB', 1, 2), ('EntA', 2, 3)]
        cp_obj.chunks_pred = chunks_pred
        cp_obj.build(config)
        
        if training:
            assert set(cp_obj.chunks) == set(chunks + chunks_pred)
            assert cp_obj.cp2label_id.size() == (num_chunks+3, num_chunks+3)
        else:
            assert set(cp_obj.chunks) == set(chunks_pred)
            assert cp_obj.cp2label_id.size() == (3, 3)
