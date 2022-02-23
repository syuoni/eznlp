# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.model.span_encoder import SpanBertLikeLayer, SpanBertLikeEncoder


def test_span_like_layer(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_layer = bert_like.encoder.layer[0]
    span_bert_like_layer = SpanBertLikeLayer(bert_layer)
    
    x = torch.randn(4, 10, 768)
    assert (span_bert_like_layer(x, x)[0] - bert_layer(x)[0]).abs().max().item() < 1e-6
    assert span_bert_like_layer(x[:, :5], x)[0].size(1) == 5
    assert span_bert_like_layer(x, x[:, :5])[0].size(1) == 10



def test_span_like_encoder(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    span_encoder = SpanBertLikeEncoder(bert_like.encoder)
    
    x_ids = torch.randint(0, 1000, size=(4, 10))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    x = bert_like.embeddings(x_ids)
    span_enc_outs = span_encoder(x, bert_outs['hidden_states'], output_query_states=True)
    assert (span_enc_outs['query_states'] - bert_outs['last_hidden_state']).abs().max().item() < 1e-6
    assert (torch.stack(span_enc_outs['all_query_states']) - torch.stack(bert_outs['hidden_states'])).abs().max().item() < 1e-6
    
    span_enc_outs = span_encoder(x[:, :5], bert_outs['hidden_states'])
    assert span_enc_outs['query_states'].size(1) == 5
