# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.model import SpanBertLikeConfig
from eznlp.training import count_params


def test_span_bert_like(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_like.eval()
    config = SpanBertLikeConfig(bert_like=bert_like, max_span_size=5)
    span_bert_like = config.instantiate()
    span_bert_like.eval()
    
    x_ids = torch.randint(0, 1000, size=(4, 10))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    all_last_query_states = span_bert_like(bert_outs['hidden_states'])
    assert len(all_last_query_states) == 4
    assert all(all_last_query_states[k].size(1) == 10-k+1 for k in range(2, 6))
    
    # Check the span representations are different from token representations
    all_hidden = [bert_outs['last_hidden_state']] + list(all_last_query_states.values())
    i = 0
    span_hidden = torch.cat([hidden[i] for hidden in all_hidden], dim=0)
    diff = (span_hidden.unsqueeze(0) - span_hidden.unsqueeze(1)).abs().sum(dim=-1)
    assert (diff > 1).sum().item() == diff.size(0) * (diff.size(0) - 1)  # 40*39 = 1560



@pytest.mark.parametrize("share_weights_int", [False, True])
@pytest.mark.parametrize("freeze", [False, True])
def test_trainble_config(share_weights_int, freeze, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    config = SpanBertLikeConfig(bert_like=bert_like, max_span_size=5, share_weights_ext=False, share_weights_int=share_weights_int, freeze=freeze)
    span_bert_like = config.instantiate()
    
    if freeze:
        assert count_params(span_bert_like) == 0
    elif share_weights_int:
        assert count_params(span_bert_like) == count_params(bert_like.encoder)
    else:
        assert count_params(span_bert_like) == count_params(bert_like.encoder)*4
