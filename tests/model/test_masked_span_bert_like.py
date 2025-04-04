# -*- coding: utf-8 -*-
import copy

import pytest
import torch
import transformers

from eznlp.model import MaskedSpanBertLikeConfig, SpanBertLikeConfig
from eznlp.training import count_params


@pytest.mark.parametrize("min_span_size", [2, 1])
@pytest.mark.parametrize("use_init_size_emb", [False, True])
def test_masked_span_bert_like(min_span_size, use_init_size_emb, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_like.eval()
    spb_config = SpanBertLikeConfig(bert_like=bert_like, min_span_size=min_span_size, max_span_size=5, use_init_size_emb=use_init_size_emb)
    spb_config.max_size_id = 3
    span_bert_like = spb_config.instantiate()
    span_bert_like.eval()

    B, L = 4, 20
    x_ids = torch.randint(0, 1000, size=(B, L))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    all_last_query_states = span_bert_like(bert_outs['hidden_states'])
    if min_span_size == 1:
        all_hidden = list(all_last_query_states.values())
    else:
        all_hidden = [bert_outs['last_hidden_state']] + list(all_last_query_states.values())


    batch_chunks = [[('EntA', 0, 1), ('EntB', 1, 2), ('EntA', 2, 3)],
                    [('EntA', 0, 20)],
                    [('EntA', 5, 10), ('EntB', 7, 10), ('EntA', 7, 8)],
                    [('EntA', 18, 20), ('EntB', 17, 19), ('EntA', 0, 3), ('EntA', 1, 2)]]

    batch_span_size_ids = [torch.tensor([end-start-1 for label, start, end in chunks]) for chunks in batch_chunks]
    batch_span_size_ids = torch.nn.utils.rnn.pad_sequence(batch_span_size_ids, batch_first=True, padding_value=0)
    batch_span_size_ids.masked_fill_(batch_span_size_ids > 3, 3)

    batch_sub_mask = torch.zeros(B, L, dtype=torch.bool)
    max_num_chunks = max(len(chunks) for chunks in batch_chunks)
    batch_ck2tok_mask = batch_sub_mask.unsqueeze(1).repeat(1, max_num_chunks, 1)
    batch_ctx2tok_mask = batch_sub_mask.unsqueeze(1).repeat(1, max_num_chunks, 1)

    for k, chunks in enumerate(batch_chunks):
        num_tokens, num_chunks = L, len(chunks)

        ck2tok_mask = torch.ones(num_chunks, num_tokens, dtype=torch.bool)
        for i, (_, start, end) in enumerate(chunks):
            for j in range(start, end):
                ck2tok_mask[i, j] = False

        # Assign to batched attention mask
        batch_ck2tok_mask[k, :num_chunks, :num_tokens].logical_or_(ck2tok_mask)
        batch_ctx2tok_mask[k, :num_chunks, :num_tokens].logical_or_(~ck2tok_mask)

    mspb_config = MaskedSpanBertLikeConfig(bert_like=bert_like, use_init_size_emb=use_init_size_emb)
    mspb_config.max_size_id = 3
    masked_span_bert_like = mspb_config.instantiate()
    if use_init_size_emb:
        masked_span_bert_like.size_embedding = copy.deepcopy(span_bert_like.size_embedding)
    masked_span_bert_like.eval()
    all_last_query_states = masked_span_bert_like(bert_outs['hidden_states'], span_size_ids=batch_span_size_ids, cp_dist_ids=None, ck2tok_mask=batch_ck2tok_mask, ctx2tok_mask=batch_ctx2tok_mask)
    span_query_hidden = all_last_query_states['span_query_state']

    # Check the masked span representations are consistent to naive span representations
    for k, chunks in enumerate(batch_chunks):
        for i, (_, start, end) in enumerate(chunks):
            # Exception for span size 1 if `min_span_size` is 2
            if (min_span_size == 1 or end-start > 1) and end-start <= 5:
                delta_hidden = span_query_hidden[k, i] - all_hidden[end-start-1][k, start]
                assert delta_hidden.abs().max().item() < 1e-5



@pytest.mark.parametrize("use_init_size_emb", [False, True])
@pytest.mark.parametrize("freeze", [False, True])
def test_trainble_config(use_init_size_emb, freeze, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    config = MaskedSpanBertLikeConfig(bert_like=bert_like, use_init_size_emb=use_init_size_emb, freeze=freeze)
    config.max_size_id = 3
    span_bert_like = config.instantiate()

    num_params = count_params(bert_like.encoder)
    if isinstance(bert_like, transformers.AlbertModel):
        num_params -= count_params(bert_like.encoder.embedding_hidden_mapping_in)

    if freeze:
        assert count_params(span_bert_like.query_bert_like) == 0
    else:
        assert count_params(span_bert_like.query_bert_like) == num_params
