# -*- coding: utf-8 -*-
import pytest
import torch
import allennlp.modules
import transformers
import flair

from eznlp.pretrained import GloVe
from eznlp.sequence_tagging.io import ConllIO
from eznlp.text_classification.io import TabularIO


@pytest.fixture
def device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def conll2003_demo():
    conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1')
    data = conll_io.read("data/conll2003/demo.eng.train")
    return data


@pytest.fixture
def yelp2013_demo():
    tabular_io = TabularIO(text_col_id=3, label_col_id=2)
    data = tabular_io.read("data/Tang2015/demo.yelp-2013-seg-20-20.dev.ss", encoding='utf-8', sep="\t\t", sentence_sep="<sssss>")
    return data


@pytest.fixture
def elmo():
    return allennlp.modules.Elmo(options_file="assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json", 
                                 weight_file="assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5", 
                                 num_output_representations=1)
    
@pytest.fixture
def bert_with_tokenizer():
    tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers/bert-base-cased")
    bert = transformers.BertModel.from_pretrained("assets/transformers/bert-base-cased")
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
    return GloVe("assets/vectors/glove.6B.100d.txt", encoding='utf-8')

