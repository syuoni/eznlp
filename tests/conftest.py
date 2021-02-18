# -*- coding: utf-8 -*-
import pytest
import torch
import allennlp.modules
import transformers
import flair

from eznlp.pretrained import GloVe
from eznlp.sequence_tagging.io import ConllIO
from eznlp.text_classification.io import TabularIO


def pytest_addoption(parser):
    parser.addoption('--device', type=str, default='auto', help="device to run tests (`auto`, `cpu` or `cuda:x`)")
    
@pytest.fixture(scope='session')
def device(request):
    device_str = request.config.getoption('--device')
    if device_str == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)
    
    
@pytest.fixture(scope='session')
def conll2003_demo():
    return ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1').read("data/conll2003/demo.eng.train")

@pytest.fixture(scope='session')
def yelp2013_demo():
    return TabularIO(text_col_id=3, label_col_id=2).read("data/Tang2015/demo.yelp-2013-seg-20-20.train.ss", 
                                                         encoding='utf-8', 
                                                         sep="\t\t", 
                                                         sentence_sep="<sssss>")


@pytest.fixture(scope='session')
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
def BERT4MLM_with_tokenizer():
    tokenizer = transformers.BertTokenizer.from_pretrained("assets/transformers/bert-base-cased")
    bert4mlm = transformers.BertForMaskedLM.from_pretrained("assets/transformers/bert-base-cased")
    return bert4mlm, tokenizer

@pytest.fixture
def RoBERTa4MLM_with_tokenizer():
    tokenizer = transformers.RobertaTokenizer.from_pretrained("assets/transformers/roberta-base")
    roberta4mlm = transformers.RobertaForMaskedLM.from_pretrained("assets/transformers/roberta-base")
    return roberta4mlm, tokenizer


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
    

@pytest.fixture(scope="module")
def a():
    return [1]

def test_a(a, device):
    import copy
    a = copy.deepcopy(a)
    a.append(2)
    print(a)
    print(device)


def test_b(a):
    print(a)
    
def test_c(a):
    print(a)


class TestClass:
    @classmethod
    def setup_class(cls):
        cls.x = [5]
    
    def _assert(self):
        self.x.append(4)
        print(self.x)
        self.y.append(9)
        print(self.y)
        assert self.a == self.b
    
    @pytest.mark.parametrize("a, b", [(1, 2), (3, 3)])
    def test_equals(self, a, b):
        self.y = [5]
        self.a = a
        self.b = b
        self._assert()
        
    def test_x(self):
        # self.y = [10]
        self._assert()
        