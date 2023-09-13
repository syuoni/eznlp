# -*- coding: utf-8 -*-
import pytest
import copy
import torch

from eznlp.model import SpanRelClassificationDecoderConfig, SpanAttrClassificationDecoderConfig
from eznlp.utils.relation import detect_inverse


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("pipeline", [True, False])
@pytest.mark.parametrize("use_inv_rel", [False, True])
@pytest.mark.parametrize("check_rht_labels", [False, True])
def test_chunk_pairs_obj(training, pipeline, use_inv_rel, check_rht_labels, EAR_data_demo):
    if pipeline:
        for entry in EAR_data_demo:
            entry['chunks_pred'] = []
    
    entry = EAR_data_demo[0]
    chunks, relations = entry['chunks'], entry['relations']
    
    config = SpanRelClassificationDecoderConfig(sym_rel_labels=['RelB'], use_inv_rel=use_inv_rel, check_rht_labels=check_rht_labels)
    config.build_vocab(EAR_data_demo)
    cp_obj = config.exemplify(entry, training=training)['cp_obj']
    
    num_chunks = len(chunks)
    assert cp_obj.relations == relations
    
    if pipeline and training:
        assert cp_obj.chunks == chunks
        assert cp_obj.non_mask.size() == (num_chunks, num_chunks)
        assert cp_obj.cp2label_id.size() == (num_chunks, num_chunks)
        if not use_inv_rel:
            assert len(list(config.enumerate_chunk_pairs(cp_obj, return_valid_only=True))) == 4 if check_rht_labels else 10
            assert cp_obj.non_mask.sum().item() == 4 if check_rht_labels else 10
            assert cp_obj.cp2label_id.sum().item() == sum(config.label2idx[label] for label, *_ in relations)
            assert all(cp_obj.cp2label_id[cp_obj.chunks.index(head), cp_obj.chunks.index(tail)] == config.label2idx[label] 
                        for label, head, tail in relations)
        else:
            assert len(list(config.enumerate_chunk_pairs(cp_obj, return_valid_only=True))) == 6 if check_rht_labels else 20
            assert cp_obj.non_mask.sum().item() == 6 if check_rht_labels else 20
            inverse_relations = detect_inverse(relations)
            assert len(inverse_relations) == 2
            assert cp_obj.cp2label_id.sum().item() == sum(config.label2idx[label] for label, *_ in relations+inverse_relations)
            assert all(cp_obj.cp2label_id[cp_obj.chunks.index(head), cp_obj.chunks.index(tail)] == config.label2idx[label] 
                        for label, head, tail in relations+inverse_relations)
        
        labels_retr = [config.idx2label[i] for i in cp_obj.cp2label_id.flatten().tolist()]
        relations_retr = [(label, head, tail) for label, (head, tail, is_valid) in zip(labels_retr, config.enumerate_chunk_pairs(cp_obj)) if is_valid and label != config.none_label]
        relations_retr = config._filter(relations_retr)
        assert set(relations_retr) == set(relations)
        
    elif pipeline and not training:
        assert len(cp_obj.chunks) == 0
        assert cp_obj.non_mask.size() == (0, 0)
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
    
    config = SpanAttrClassificationDecoderConfig()
    assert config.multilabel
    config.build_vocab(EAR_data_demo)
    cs_obj = config.exemplify(entry, training=training)['cs_obj']
    
    num_chunks = len(chunks)
    assert cs_obj.attributes == attributes
    
    if pipeline and training:
        assert cs_obj.chunks == chunks
        assert cs_obj.cs2label_id.size() == (num_chunks, config.voc_dim)
        assert cs_obj.cs2label_id[:, 1:].sum().item() == len(attributes)
        assert (cs_obj.cs2label_id.sum(dim=0) >= 1).all().item()
        assert all(cs_obj.cs2label_id[cs_obj.chunks.index(chunk), config.label2idx[label]] == 1 for label, chunk in attributes)
        
        all_confidences = copy.deepcopy(cs_obj.cs2label_id)
        all_confidences[all_confidences[:,config.none_idx] > (1-config.conf_thresh)] = 0
        all_confidences[:,config.none_idx] = 0
        pos_entries = torch.nonzero(all_confidences > config.conf_thresh).cpu().tolist()
        attributes_retr = [(config.idx2label[i], cs_obj.chunks[cidx]) for cidx, i in pos_entries]
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
