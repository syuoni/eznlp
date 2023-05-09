# -*- coding: utf-8 -*-
from eznlp.nn.init import reinit_bert_like_


def test_reinit_bert_like(bert_with_tokenizer):
    bert, tokenizer = bert_with_tokenizer
    
    assert abs(bert.encoder.layer[0].attention.self.query.weight.std().item() - 0.0344940721988678) < 1e-6
    reinit_bert_like_(bert)
    assert abs(bert.encoder.layer[0].attention.self.query.weight.std().item() - 0.0176) < 1e-3
