# -*- coding: utf-8 -*-
import pytest
import torch
import transformers

from eznlp.model import SpanBertLikeConfig
from eznlp.training import count_params


@pytest.mark.parametrize("min_span_size", [2, 1])
@pytest.mark.parametrize("use_init_size_emb", [False, True])
def test_span_bert_like(min_span_size, use_init_size_emb, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_like.eval()
    config = SpanBertLikeConfig(bert_like=bert_like, min_span_size=min_span_size, max_span_size=5, use_init_size_emb=use_init_size_emb)
    config.max_size_id = 3
    span_bert_like = config.instantiate()
    span_bert_like.eval()
    
    x_ids = torch.randint(0, 1000, size=(4, 10))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    all_last_query_states = span_bert_like(bert_outs['hidden_states'])
    assert len(all_last_query_states) == 6-min_span_size
    assert all(all_last_query_states[k].size(1) == 10-k+1 for k in range(min_span_size, 6))
    
    # Check the span representations are different from token representations
    if min_span_size == 1:
        all_hidden = list(all_last_query_states.values())
    else:
        all_hidden = [bert_outs['last_hidden_state']] + list(all_last_query_states.values())
    
    # The first example in the batch; there are totally 40 (=10+9+8+7+6) span representations
    i = 0
    span_hidden = torch.cat([hidden[i] for hidden in all_hidden], dim=0)
    diff = (span_hidden.unsqueeze(0) - span_hidden.unsqueeze(1)).abs().sum(dim=-1)
    assert (diff > 1).sum().item() == diff.size(0) * (diff.size(0) - 1)  # 40*39 = 1560



@pytest.mark.parametrize("min_span_size", [2, 1])
@pytest.mark.parametrize("use_init_size_emb", [False, True])
@pytest.mark.parametrize("share_weights_int", [False, True])
@pytest.mark.parametrize("freeze", [False, True])
def test_trainble_config(min_span_size, use_init_size_emb, share_weights_int, freeze, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    config = SpanBertLikeConfig(bert_like=bert_like, min_span_size=min_span_size, max_span_size=5, use_init_size_emb=use_init_size_emb, share_weights_ext=False, share_weights_int=share_weights_int, freeze=freeze)
    config.max_size_id = 3
    span_bert_like = config.instantiate()
    
    num_params = count_params(bert_like.encoder)
    if isinstance(bert_like, transformers.AlbertModel):
        num_params -= count_params(bert_like.encoder.embedding_hidden_mapping_in)
    
    if freeze:
        assert count_params(span_bert_like.query_bert_like) == 0
    elif share_weights_int:
        assert count_params(span_bert_like.query_bert_like) == num_params
    else:
        assert count_params(span_bert_like.query_bert_like) == num_params*(6-min_span_size)
