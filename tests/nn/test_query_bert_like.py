# -*- coding: utf-8 -*-
import torch
import transformers

from eznlp.nn.modules.query_bert_like import QueryBertLikeLayer, QueryAlbertLayer, QueryBertLikeEncoder


def test_query_bert_like_layer(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    if isinstance(bert_like, transformers.AlbertModel):
        bert_layer = bert_like.encoder.albert_layer_groups[0].albert_layers[0]
        query_bert_like_layer = QueryAlbertLayer(bert_layer)
    else:
        bert_layer = bert_like.encoder.layer[0]
        query_bert_like_layer = QueryBertLikeLayer(bert_layer)
    
    x = torch.randn(4, 10, 768)
    assert (query_bert_like_layer(x, x)[0] - bert_layer(x)[0]).abs().max().item() < 1e-6
    assert query_bert_like_layer(x[:, :5], x)[0].size(1) == 5
    assert query_bert_like_layer(x, x[:, :5])[0].size(1) == 10



def test_query_bert_like_encoder(bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    query_encoder = QueryBertLikeEncoder(bert_like.encoder)
    
    x_ids = torch.randint(0, 1000, size=(4, 10))
    bert_outs = bert_like(x_ids, output_hidden_states=True)
    x = bert_like.embeddings(x_ids)
    if isinstance(bert_like, transformers.AlbertModel):
        x = bert_like.encoder.embedding_hidden_mapping_in(x)
    query_enc_outs = query_encoder(x, bert_outs['hidden_states'], output_query_states=True)
    assert (query_enc_outs['last_query_state'] - bert_outs['last_hidden_state']).abs().max().item() < 1e-6
    assert (torch.stack(query_enc_outs['query_states']) - torch.stack(bert_outs['hidden_states'])).abs().max().item() < 1e-6
    
    query_enc_outs = query_encoder(x[:, :5], bert_outs['hidden_states'])
    assert query_enc_outs['last_query_state'].size(1) == 5
