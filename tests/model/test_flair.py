# -*- coding: utf-8 -*-
import pytest
import os
import torch
import flair

from eznlp.token import TokenSequence
from eznlp.model import FlairConfig
from eznlp.training import count_params


@pytest.mark.parametrize("agg_mode", ['last', 'mean'])
def test_flair_embeddings(agg_mode, flair_lm):
    batch_tokenized_text = [["I", "like", "it", "."], 
                            ["Do", "you", "love", "me", "?"], 
                            ["Sure", "!"], 
                            ["Future", "it", "out"]]
    
    flair_emb = flair.embeddings.FlairEmbeddings(flair_lm)
    flair_sentences = [flair.data.Sentence(" ".join(sent), use_tokenizer=False) for sent in batch_tokenized_text]
    flair_emb.embed(flair_sentences)
    expected = torch.nn.utils.rnn.pad_sequence([torch.stack([tok.embedding for tok in sent]) for sent in flair_sentences], 
                                               batch_first=True, 
                                               padding_value=0.0)
    
    flair_config = FlairConfig(flair_lm=flair_lm, agg_mode=agg_mode)
    flair_embedder = flair_config.instantiate()
    
    
    batch_tokens = [TokenSequence.from_tokenized_text(tokenized_text) for tokenized_text in batch_tokenized_text]
    batch_flair_ins = flair_config.batchify([flair_config.exemplify(tokens) for tokens in batch_tokens])
    if agg_mode.lower() == 'last':
        assert (flair_embedder(**batch_flair_ins) == expected).all().item()
    else:
        assert (flair_embedder(**batch_flair_ins) != expected).any().item()



@pytest.mark.parametrize("freeze", [True, False])
@pytest.mark.parametrize("use_gamma", [True, False])
def test_trainble_config(freeze, use_gamma, flair_lm):
    flair_config = FlairConfig(flair_lm=flair_lm, freeze=freeze, use_gamma=use_gamma)
    flair_embedder = flair_config.instantiate()
    
    expected_num_trainable_params = 0
    if not freeze:
        expected_num_trainable_params += count_params(flair_lm, return_trainable=False)
    if use_gamma:
        expected_num_trainable_params += 1
    assert count_params(flair_embedder) == expected_num_trainable_params



def test_serialization(flair_fw_lm):
    config = FlairConfig(flair_lm=flair_fw_lm)
    
    config_path = "cache/flair_embedder.config"
    torch.save(config, config_path)
    assert os.path.getsize(config_path) < 1024 * 1024  # 1MB
