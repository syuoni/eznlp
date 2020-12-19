# -*- coding: utf-8 -*-
import pytest
import torch
from torchtext.experimental.vectors import GloVe
import allennlp.modules
import transformers
import flair


@pytest.fixture
def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def elmo():
    options_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json"
    weight_file = "assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
    return allennlp.modules.elmo.Elmo(options_file, weight_file, num_output_representations=1)
    

@pytest.fixture
def bert_with_tokenizer():
    tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers_cache/bert-base-cased")
    bert = transformers.BertModel.from_pretrained("assets/transformers_cache/bert-base-cased")
    return bert, tokenizer


@pytest.fixture
def flair_fw_lm():
    return flair.models.LanguageModel.load_language_model("assets/flair/lm-mix-english-forward-v0.2rc.pt")

@pytest.fixture
def flair_bw_lm():
    return flair.models.LanguageModel.load_language_model("assets/flair/lm-mix-english-backward-v0.2rc.pt")


@pytest.fixture
def glove100():
    # https://nlp.stanford.edu/projects/glove/
    return GloVe(name='6B', dim=100, root="assets/vector_cache", validate_file=False)

