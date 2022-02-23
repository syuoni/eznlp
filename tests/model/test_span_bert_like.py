# -*- coding: utf-8 -*-
import torch

from eznlp.model import SpanBertLikeConfig


def test_span_bert_like(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    config = SpanBertLikeConfig(bert_like=bert_like, max_span_size=5)
    span_bert_like = config.instantiate()
    
    x_ids = torch.randint(0, 1000, size=(4, 10))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    all_last_query_states = span_bert_like(bert_outs['hidden_states'])
    assert len(all_last_query_states) == 4
    assert all(all_last_query_states[k].size(1) == 10-k+1 for k in range(2, 6))
