# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.model import BoundarySelectionDecoderConfig, SpecificSpanRelClsDecoderConfig
from eznlp.model.decoder.boundaries import _spans_from_upper_triangular, _spans_from_diagonals, _span_pairs_from_diagonals
from eznlp.model.decoder.boundaries import _span2diagonal, _diagonal2span


@pytest.mark.parametrize("sb_epsilon", [0.0, 0.1])
@pytest.mark.parametrize("sl_epsilon", [0.0, 0.1])
def test_boundaries_obj(sb_epsilon, sl_epsilon, EAR_data_demo):
    entry = EAR_data_demo[0]
    tokens, chunks = entry['tokens'], entry['chunks']
    
    config = BoundarySelectionDecoderConfig(sb_epsilon=sb_epsilon, sl_epsilon=sl_epsilon)
    config.build_vocab(EAR_data_demo)
    boundaries_obj = config.exemplify(entry)['boundaries_obj']
    
    num_tokens, num_chunks = len(tokens), len(chunks)
    assert boundaries_obj.chunks == chunks
    if sb_epsilon == 0 and sl_epsilon == 0:
        assert all(boundaries_obj.boundary2label_id[start, end-1] == config.label2idx[label] for label, start, end in chunks)
        labels_retr = [config.idx2label[i] for i in boundaries_obj.boundary2label_id[torch.arange(num_tokens) >= torch.arange(num_tokens).unsqueeze(-1)].tolist()]
    else:
        assert all(boundaries_obj.boundary2label_id[start, end-1].argmax() == config.label2idx[label] for label, start, end in chunks)
        labels_retr = [config.idx2label[i] for i in boundaries_obj.boundary2label_id[torch.arange(num_tokens) >= torch.arange(num_tokens).unsqueeze(-1)].argmax(dim=-1).tolist()]
        
        assert (boundaries_obj.boundary2label_id.sum(dim=-1) - 1).abs().max().item() < 1e-6
        if sb_epsilon == 0:
            assert (boundaries_obj.boundary2label_id[:, :, config.none_idx] < 1).sum().item() == num_chunks
        else:
            assert (boundaries_obj.boundary2label_id[:, :, config.none_idx] < 1).sum().item() > num_chunks
    
    chunks_retr = [(label, start, end) for label, (start, end) in zip(labels_retr, _spans_from_upper_triangular(num_tokens)) if label != config.none_label]
    assert set(chunks_retr) == set(chunks)
    
    config.in_dim = 200
    decoder = config.instantiate()
    # \sum_{k=0}^N k(N-k), where N is `num_tokens`
    assert decoder._span_size_ids.sum().item() == (num_tokens**3 - num_tokens) // 6
    assert decoder._span_non_mask.sum().item() == (num_tokens**2 + num_tokens) // 2



@pytest.mark.parametrize("sb_epsilon", [0.1, 0.5])
@pytest.mark.parametrize("sb_size", [1, 2, 3])
def test_boundaries_obj_for_boundary_smoothing(sb_epsilon, sb_size):
    entry = {'tokens': list("abcdef"), 
             'chunks': [('EntA', 0, 1), ('EntA', 0, 4), ('EntB', 0, 5), ('EntA', 3, 5), ('EntA', 4, 5)]}
    config = BoundarySelectionDecoderConfig(sb_epsilon=sb_epsilon, sb_size=sb_size)
    config.build_vocab([entry])
    boundaries_obj = config.exemplify(entry)['boundaries_obj']
    
    num_tokens, num_chunks = len(entry['tokens']), len(entry['chunks'])
    span_sizes = torch.arange(num_tokens) - torch.arange(num_tokens).unsqueeze(-1) + 1
    assert (boundaries_obj.boundary2label_id.sum(dim=-1) - 1).abs().max().item() < 1e-6
    assert (boundaries_obj.boundary2label_id[:, :, 1:].sum() - num_chunks).abs().max().item() < 1e-6
    assert (boundaries_obj.boundary2label_id[span_sizes<=0] - torch.tensor([1.0, 0.0, 0.0])).abs().max().item() < 1e-6
    
    if sb_size == 1:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(1/2)*sb_epsilon, 1-(3/4)*sb_epsilon, (1/4)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(1/2)*sb_epsilon, (1/4)*sb_epsilon, 1-(3/4)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(3/4)*sb_epsilon, 1-(3/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
    elif sb_size == 2:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(9/16)*sb_epsilon, 1-(11/16)*sb_epsilon, (1/8)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(1/2)*sb_epsilon, (1/8)*sb_epsilon, 1-(5/8)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(5/8)*sb_epsilon, 1-(5/8)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(3/8)*sb_epsilon, 1-(3/8)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
    elif sb_size == 3:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(7/36)*sb_epsilon, 1-(7/36)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(37/72)*sb_epsilon, 1-(43/72)*sb_epsilon, (1/12)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(4/9)*sb_epsilon, (1/9)*sb_epsilon, 1-(5/9)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(19/36)*sb_epsilon, 1-(5/9)*sb_epsilon, (1/36)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(1/3)*sb_epsilon, 1-(1/3)*sb_epsilon, 0.0])).abs().max().item() < 1e-6



@pytest.mark.parametrize("neg_sampling_rate", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("neg_sampling_surr_rate", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("training", [True, False])
def test_boundaries_obj_for_neg_sampling(neg_sampling_rate, neg_sampling_surr_rate, training):
    entry = {'tokens': list("abcdefhijk"), 
             'chunks': [('EntA', 0, 1), ('EntA', 0, 4), ('EntB', 0, 5), ('EntA', 3, 5), ('EntA', 4, 5)]}
    config = BoundarySelectionDecoderConfig(neg_sampling_rate=neg_sampling_rate, 
                                            neg_sampling_surr_rate=neg_sampling_surr_rate, 
                                            neg_sampling_surr_size=3)
    config.build_vocab([entry])
    boundaries_obj = config.exemplify(entry, training=training)['boundaries_obj']
    
    if (not training) or (neg_sampling_rate == 1):
        assert not hasattr(boundaries_obj, 'non_mask')
    elif neg_sampling_rate ==0 and neg_sampling_surr_rate == 0:
        assert boundaries_obj.non_mask.sum().item() == 5
    elif neg_sampling_rate ==0 and neg_sampling_surr_rate == 1:
        assert boundaries_obj.non_mask.sum().item() == 30
    else:
        assert abs(boundaries_obj.non_mask.sum().item() - (5 + 50*neg_sampling_rate + 25*(1-neg_sampling_rate)*neg_sampling_surr_rate)) < 10



@pytest.mark.parametrize("training", [True, False])
def test_diag_boundaries_pair_obj(training, EAR_data_demo):
    entry = EAR_data_demo[0]
    entry['chunks_pred'] = []
    tokens, chunks, relations = entry['tokens'], entry['chunks'], entry['relations']
    
    config = SpecificSpanRelClsDecoderConfig(max_span_size=3)
    config.build_vocab(EAR_data_demo)
    dbp_obj = config.exemplify(entry, training=training)['dbp_obj']
    
    num_tokens = len(tokens)
    num_spans = (num_tokens-1) * 3  # max_span_size=3
    assert dbp_obj.chunks == (chunks if training else [])
    assert dbp_obj.relations == relations
    assert dbp_obj.dbp2label_id.size() == (num_spans, num_spans)
    
    assert dbp_obj.dbp2label_id.sum() == sum(config.label2idx[label] for label, *_ in relations)
    assert all(dbp_obj.dbp2label_id[_span2diagonal(h_start, h_end, num_tokens), _span2diagonal(t_start, t_end, num_tokens)] == config.label2idx[label] 
                   for label, (_, h_start, h_end), (_, t_start, t_end) in relations)
    
    labels_retr = [config.idx2label[i] for i in dbp_obj.dbp2label_id.flatten().tolist()]
    relations_retr = []
    for label, ((h_start, h_end), (t_start, t_end)) in zip(labels_retr, _span_pairs_from_diagonals(num_tokens, 3)):
        head = (dbp_obj.span2ck_label.get((h_start, h_end), config.ck_none_label), h_start, h_end)
        tail = (dbp_obj.span2ck_label.get((t_start, t_end), config.ck_none_label), t_start, t_end)
        if label != config.none_label and head[0] != config.ck_none_label and tail[0] != config.ck_none_label:
            relations_retr.append((label, head, tail))
    if training:
        assert set(relations_retr) == set(relations)
    else:
        assert len(relations_retr) == 0  # `dbp_obj.chunks` is empty 



@pytest.mark.parametrize("seq_len", [1, 5, 10, 100])
def test_spans_from_upper_triangular(seq_len):
    assert len(list(_spans_from_upper_triangular(seq_len))) == (seq_len+1)*seq_len // 2


@pytest.mark.parametrize("seq_len, max_span_size", [(5, 1),   (5, 5), 
                                                    (10, 1),  (10, 5), (10, 10), 
                                                    (100, 1), (100, 10), (100, 100)])
def test_spans_from_diagonals(seq_len, max_span_size):
    assert len(list(_spans_from_diagonals(seq_len, max_span_size))) == (seq_len*2-max_span_size+1)*max_span_size // 2


@pytest.mark.parametrize("seq_len", [5, 10, 100])
def test_spans_from_functions(seq_len):
    assert set(_spans_from_diagonals(seq_len)) == set(_spans_from_upper_triangular(seq_len))


@pytest.mark.parametrize("seq_len", [5, 10, 100])
def test_span2diagonal(seq_len):
    num_spans = (seq_len+1)*seq_len // 2
    assert [_span2diagonal(start, end, seq_len) for start, end in _spans_from_diagonals(seq_len)] == list(range(num_spans))
    assert [_diagonal2span(k, seq_len) for k in range(num_spans)] == list(_spans_from_diagonals(seq_len))
