# -*- coding: utf-8 -*-
import pytest
import spacy
import jieba
import torch
import allennlp.modules
import transformers
import flair

from eznlp import auto_device
from eznlp.token import TokenSequence
from eznlp.vectors import Vectors, GloVe
from eznlp.io import TabularIO, ConllIO, JsonIO, BratIO


def pytest_addoption(parser):
    parser.addoption('--device', type=str, default='auto', help="device to run tests (`auto`, `cpu` or `cuda:x`)")
    parser.addoption('--runslow', default=False, action='store_true', help="whether to run slow tests")
    
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    
@pytest.fixture(scope='session')
def device(request):
    device_str = request.config.getoption('--device')
    if device_str == 'auto':
        return auto_device()
    else:
        return torch.device(device_str)
    
    
@pytest.fixture
def spacy_nlp_en():
    return spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner'])


@pytest.fixture
def conll2003_demo():
    return ConllIO(text_col_id=0, 
                   tag_col_id=3, 
                   scheme='BIO1', 
                   verbose=False).read("data/conll2003/demo.eng.train")

@pytest.fixture
def ResumeNER_demo():
    return ConllIO(text_col_id=0, 
                   tag_col_id=1, 
                   scheme='BMES', 
                   encoding='utf-8', 
                   token_sep="", 
                   pad_token="", 
                   verbose=False).read("data/ResumeNER/demo.train.char.bmes")

@pytest.fixture
def yelp_full_demo(spacy_nlp_en):
    return TabularIO(text_col_id=1, 
                     label_col_id=0, 
                     sep=",", 
                     mapping={"\\n": "\n", '\\"': '"'}, 
                     tokenize_callback=spacy_nlp_en, 
                     case_mode='lower', 
                     verbose=False).read("data/yelp_review_full/demo.train.csv")

@pytest.fixture
def conll2004_demo():
    return JsonIO(text_key='tokens', 
                  chunk_key='entities', 
                  chunk_type_key='type', 
                  chunk_start_key='start', 
                  chunk_end_key='end', 
                  relation_key='relations', 
                  relation_type_key='type', 
                  relation_head_key='head', 
                  relation_tail_key='tail', 
                  verbose=False).read("data/conll2004/demo.conll04_train.json")


@pytest.fixture
def HwaMei_demo():
    return BratIO(tokenize_callback='char', 
                  has_ins_space=True, 
                  ins_space_tokenize_callback=jieba.cut, 
                  parse_attrs=True, 
                  parse_relations=True, 
                  line_sep="\r\n", 
                  max_len=200, 
                  encoding='utf-8', 
                  token_sep="", 
                  pad_token="").read("data/HwaMei/demo.ChaFangJiLu.txt")


@pytest.fixture
def re_data_demo():
    tokenized_raw_text = ["This", "is", "a", "-3.14", "demo", ".", 
                          "Those", "are", "an", "APPLE", "and", "some", "glass", "bottles", ".", 
                          "This", "is", "a", "very", "very", "very", "very", "very", "long", "entity", "?"]
    tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
    chunks = [('EntA', 4, 5), ('EntA', 9, 10), ('EntB', 12, 14), ('EntC', 18, 25)]
    relations = [('RelA', chunks[0], chunks[1]), 
                 ('RelA', chunks[0], chunks[2]), 
                 ('RelB', chunks[1], chunks[2]), 
                 ('RelB', chunks[2], chunks[1])]
    return [{'tokens': tokens, 'chunks': chunks, 'relations': relations}]


@pytest.fixture
def glove100():
    return GloVe("assets/vectors/glove.6B.100d.txt", encoding='utf-8')

@pytest.fixture
def ctb50():
    return Vectors.load("assets/vectors/ctb.50d.vec", encoding='utf-8')

@pytest.fixture
def elmo():
    return allennlp.modules.Elmo(options_file="assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_options.json", 
                                 weight_file="assets/allennlp/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5", 
                                 num_output_representations=1)


BERT_PATH = "assets/transformers/bert-base-cased"
ROBERTA_PATH = "assets/transformers/roberta-base"

@pytest.fixture
def bert_with_tokenizer():
    return (transformers.BertModel.from_pretrained(BERT_PATH), 
            transformers.BertTokenizer.from_pretrained(BERT_PATH))

@pytest.fixture
def bert4mlm_with_tokenizer():
    return (transformers.BertForMaskedLM.from_pretrained(BERT_PATH), 
            transformers.BertTokenizer.from_pretrained(BERT_PATH))

@pytest.fixture
def roberta_with_tokenizer():
    """
    The GPT-2/RoBERTa tokenizer expects a space before all the words. 
    https://github.com/huggingface/transformers/issues/1196
    https://github.com/pytorch/fairseq/blob/master/fairseq/models/roberta/hub_interface.py#L38-L56
    """
    return (transformers.RobertaModel.from_pretrained(ROBERTA_PATH), 
            transformers.RobertaTokenizer.from_pretrained(ROBERTA_PATH, add_prefix_space=True))

@pytest.fixture
def roberta4mlm_with_tokenizer():
    return (transformers.RobertaForMaskedLM.from_pretrained(ROBERTA_PATH), 
            transformers.RobertaTokenizer.from_pretrained(ROBERTA_PATH, add_prefix_space=True))

@pytest.fixture(params=['bert', 'roberta'])
def bert_like_with_tokenizer(request, bert_with_tokenizer, roberta_with_tokenizer):
    if request.param == 'bert':
        return bert_with_tokenizer
    elif request.param == 'roberta':
        return roberta_with_tokenizer

@pytest.fixture(params=['bert', 'roberta'])
def bert_like4mlm_with_tokenizer(request, bert4mlm_with_tokenizer, roberta4mlm_with_tokenizer):
    if request.param == 'bert':
        return bert4mlm_with_tokenizer
    elif request.param == 'roberta':
        return roberta4mlm_with_tokenizer


@pytest.fixture
def flair_fw_lm():
    return flair.models.LanguageModel.load_language_model("assets/flair/lm-mix-english-forward-v0.2rc.pt")

@pytest.fixture
def flair_bw_lm():
    return flair.models.LanguageModel.load_language_model("assets/flair/lm-mix-english-backward-v0.2rc.pt")

@pytest.fixture(params=['fw', 'bw'])
def flair_lm(request, flair_fw_lm, flair_bw_lm):
    if request.param == 'fw':
        return flair_fw_lm
    elif request.param == 'bw':
        return flair_bw_lm
    