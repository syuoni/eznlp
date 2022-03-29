# -*- coding: utf-8 -*-
import pytest
import itertools

from eznlp.model import SpanRelClassificationDecoderConfig, SpanAttrClassificationDecoderConfig


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("pipeline", [True, False])
def test_chunk_pairs_obj(training, pipeline, EAR_data_demo):
    if pipeline:
        for entry in EAR_data_demo:
            entry['chunks_pred'] = []
    
    entry = EAR_data_demo[0]
    chunks, relations = entry['chunks'], entry['relations']
    
    config = SpanRelClassificationDecoderConfig(max_span_size=3)
    config.build_vocab(EAR_data_demo)
    cp_obj = config.exemplify(entry, training=training)['cp_obj']
    
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



@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("pipeline", [True, False])
def test_chunk_singles_obj(training, pipeline, EAR_data_demo):
    if pipeline:
        for entry in EAR_data_demo:
            entry['chunks_pred'] = []
    
    entry = EAR_data_demo[0]
    chunks, attributes = entry['chunks'], entry['attributes']
    
    config = SpanAttrClassificationDecoderConfig(max_span_size=3)
    config.build_vocab(EAR_data_demo)
    cs_obj = config.exemplify(entry, training=training)['cs_obj']
    
    num_chunks = len(chunks)
    assert cs_obj.attributes == attributes
    
    if pipeline and training:
        assert cs_obj.chunks == chunks
        assert cs_obj.cs2label_id.size() == (num_chunks, config.voc_dim)
        assert cs_obj.cs2label_id[:, 1:].sum() == len(attributes)
        assert (cs_obj.cs2label_id.sum(dim=0) >= 1).all().item()
        assert all(cs_obj.cs2label_id[cs_obj.chunks.index(chunk), config.label2idx[label]] == 1 for label, chunk in attributes)
        
        attributes_retr = []
        for chunk, ck_confidences in zip(cs_obj.chunks, cs_obj.cs2label_id):
            labels_retr = [config.idx2label[i] for i, c in enumerate(ck_confidences.tolist()) if c >= config.confidence_threshold]
            if config.none_label not in labels_retr:
                attributes_retr.extend([(label, chunk) for label in labels_retr])
        assert set(attributes_retr) == set(attributes)
        
    elif pipeline and not training:
        assert len(cs_obj.chunks) == 0
        assert cs_obj.cs2label_id.size() == (0, config.voc_dim)
        
    else:
        assert not hasattr(cs_obj, 'chunks')
        assert not hasattr(cs_obj, 'cs2label_id')
        
        chunks_pred = [('EntA', 0, 1), ('EntB', 1, 2), ('EntA', 2, 3)]
        cs_obj.chunks_pred = chunks_pred
        cs_obj.build(config)
        
        if training:
            assert set(cs_obj.chunks) == set(chunks + chunks_pred)
            assert cs_obj.cs2label_id.size() == (num_chunks+3, config.voc_dim)
        else:
            assert set(cs_obj.chunks) == set(chunks_pred)
            assert cs_obj.cs2label_id.size() == (3, config.voc_dim)
