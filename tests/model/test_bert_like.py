# -*- coding: utf-8 -*-
import pytest
import os
import string
import random
import numpy
import pandas
import torch

from eznlp.token import TokenSequence
from eznlp.model import BertLikeConfig
from eznlp.model.bert_like import (truecase_for_bert_like, 
                                   truncate_for_bert_like, 
                                   segment_uniformly_for_bert_like, 
                                   subtokenize_for_bert_like, 
                                   _tokenized2nested)
from eznlp.training import count_params
from eznlp.io import TabularIO



@pytest.mark.parametrize("mix_layers", ['trainable', 'top'])
@pytest.mark.parametrize("use_gamma", [True])
@pytest.mark.parametrize("freeze", [True])
def test_trainble_config(mix_layers, use_gamma, freeze, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_like_config = BertLikeConfig(bert_like=bert_like, tokenizer=tokenizer, 
                                      freeze=freeze, mix_layers=mix_layers, use_gamma=use_gamma)
    bert_like_embedder = bert_like_config.instantiate()
    
    expected_num_trainable_params = 0
    if not freeze:
        expected_num_trainable_params += count_params(bert_like, return_trainable=False)
    if mix_layers.lower() == 'trainable':
        expected_num_trainable_params += 13
    if use_gamma:
        expected_num_trainable_params += 1
    
    assert count_params(bert_like_embedder) == expected_num_trainable_params



@pytest.mark.parametrize("paired_inputs", [True, False])
def test_paired_inputs_config(paired_inputs, bert_like_with_tokenizer):
    bert_like, tokenizer = bert_like_with_tokenizer
    bert_like_config = BertLikeConfig(bert_like=bert_like, tokenizer=tokenizer, paired_inputs=paired_inputs)
    
    tokens = TokenSequence.from_tokenized_text([c for c in random.choices(string.ascii_letters, k=100)])
    entry = {'tokens': tokens + TokenSequence.from_tokenized_text([tokenizer.sep_token]) + tokens}
    example = bert_like_config.exemplify(entry['tokens'])
    
    if paired_inputs:
        assert 'sub_tok_type_ids' in example
        assert example['sub_tok_type_ids'].sum().item() == 101
    else:
        assert 'sub_tok_type_ids' not in example



def test_serialization(bert_with_tokenizer):
    bert, tokenizer = bert_with_tokenizer
    config = BertLikeConfig(tokenizer=tokenizer, bert_like=bert)
    
    config_path = "cache/bert_embedder.config"
    torch.save(config, config_path)
    assert os.path.getsize(config_path) < 1024 * 1024  # 1MB



def test_truecase_for_bert_like():
    tokenized_raw_text = ['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']
    tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
    data = [{'tokens': tokens}]
    new_data = truecase_for_bert_like(data, verbose=False)
    assert new_data[0]['tokens'].raw_text == ['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']



@pytest.mark.slow
@pytest.mark.parametrize("mode", ["head+tail", "head-only", "tail-only"])
def test_truncate_for_bert_like(mode, bert_with_tokenizer):
    bert, tokenizer = bert_with_tokenizer
    max_len = tokenizer.model_max_length - 2
    
    tabular_io = TabularIO(text_col_id=3, label_col_id=2, sep="\t\t", mapping={"<sssss>": "\n"}, encoding='utf-8', case_mode='lower')
    data = tabular_io.read("data/Tang2015/yelp-2013-seg-20-20.test.ss")
    data = [data_entry for data_entry in data if len(data_entry['tokens']) >= max_len-10]
    assert max(len(data_entry['tokens']) for data_entry in data) > max_len
    
    data = truncate_for_bert_like(data, tokenizer, mode)
    sub_lens = [sum(len(word) for word in _tokenized2nested(data_entry['tokens'].raw_text, tokenizer)) for data_entry in data]
    sub_lens = pandas.Series(sub_lens)
    
    assert (sub_lens <= max_len).all()
    assert (sub_lens == max_len).sum() >= (len(sub_lens) / 2)



@pytest.mark.parametrize("token_len", [1, 2, 5])
@pytest.mark.parametrize("max_len", [50, 120])
def test_segment_uniformly_for_bert_like(bert_with_tokenizer, token_len, max_len):
    bert, tokenizer = bert_with_tokenizer
    tokenizer.model_max_length = max_len + 2
    
    tokens = TokenSequence.from_tokenized_text([c*token_len for c in random.choices(string.ascii_letters, k=100)])
    chunks = [('EntA', k*10, k*10+5) for k in range(10)]
    data = [{'tokens': tokens, 'chunks': chunks}]
    new_data = segment_uniformly_for_bert_like(data, tokenizer, verbose=False)
    
    assert all(len(entry['tokens']) <= max_len for entry in new_data)
    assert all(0 <= start and end <= len(entry['tokens']) for entry in new_data for label, start, end in entry['chunks'])
    
    chunk_texts = [tokens[start:end].raw_text for label, start, end in chunks]
    chunk_texts_retr = [entry['tokens'][start:end].raw_text for entry in new_data for label, start, end in entry['chunks']]
    assert chunk_texts_retr == chunk_texts
    
    span_starts = [0] + numpy.cumsum([len(entry['tokens']) for entry in new_data]).tolist()
    chunks_retr = [(label, span_start+start, span_start+end) for entry, span_start in zip(new_data, span_starts) for label, start, end in entry['chunks']]
    assert chunks_retr == chunks



def test_subtokenize_for_bert_like(bert_with_tokenizer):
    bert, tokenizer = bert_with_tokenizer
    
    tokens = TokenSequence.from_tokenized_text(["I", "like", 'it', 'sooooo', 'much!'])
    chunks = [('EntA', start, end) for start in range(len(tokens)) for end in range(start+1, len(tokens)+1)]
    data = [{'tokens': tokens, 'chunks': chunks}]
    new_data = subtokenize_for_bert_like(data, tokenizer, verbose=False)
    
    assert new_data[0]['sub2ori_idx'] == [0, 1, 2, 3, 3.333, 3.667, 4, 4.5, 5]
    assert new_data[0]['ori2sub_idx'] == [0, 1, 2, 3, 6, 8]
    for entry, new_entry in zip(data, new_data):
        for ck, new_ck in zip(entry['chunks'], new_entry['chunks']):
            assert "".join(entry['tokens'].raw_text[ck[1]:ck[2]]).lower() == "".join(new_entry['tokens'].raw_text[new_ck[1]:new_ck[2]]).replace("##", "").lower()


def test_subtokenize_for_bert_like_with_data(bert_like_with_tokenizer, conll2003_demo):
    bert_like, tokenizer = bert_like_with_tokenizer
    
    conll2003_demo = conll2003_demo[1:]  # The fisrt sentence include "-DOCSTART-"
    for entry in conll2003_demo:
        entry['chunks'] = [('EntA', start, end) for start in range(len(entry['tokens'])) for end in range(start+1, len(entry['tokens'])+1)]
    new_data = subtokenize_for_bert_like(conll2003_demo, tokenizer, verbose=False)
    
    for entry, new_entry in zip(conll2003_demo, new_data):
        for ck, new_ck in zip(entry['chunks'], new_entry['chunks']):
            assert ("".join(entry['tokens'].raw_text[ck[1]:ck[2]]).lower() == 
                    "".join(new_entry['tokens'].raw_text[new_ck[1]:new_ck[2]]).replace("##", "").replace("Ġ", "").lower())
