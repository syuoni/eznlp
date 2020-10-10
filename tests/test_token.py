# -*- coding: utf-8 -*-
from eznlp.token import Full2Half
from eznlp import Token, TokenSequence


def test_full2half():
    assert Full2Half.full2half("，；：？！") == ",;:?!"
    assert Full2Half.half2full(",;:?!") == "，；：？！"
    
    
class TestToken(object):
    def test_assign_attr(self):
        tok = Token("-5.44", chunking='B-NP')
        assert hasattr(tok, 'chunking')
        assert tok.chunking == 'B-NP'
    
    def test_adaptive_lower(self):
        tok = Token("Of")
        assert tok.text == "of"
        
        tok = Token("THE")
        assert tok.text == "the"
        
        tok = Token("marry")
        assert tok.text == "marry"
        
        tok = Token("MARRY")
        assert tok.text == "MARRY"
        
        tok = Token("WATERMELON")
        assert tok.text == "watermelon"
        
        tok = Token("WATERMELON-")
        assert tok.text == "WATERMELON-"
        
        tok = Token("Jack")
        assert tok.text == "jack"
        
        tok = Token("jACK")
        assert tok.text == "jACK"
        
    def test_en_shapes(self):
        tok = Token("Jack")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': True,
               'any_digit': False,
               'any_punct': False,
               'init_upper': True,
               'init_lower': False,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': False,
               'any_noninit_lower': True,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': True,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        assert all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names)
        
        
        tok = Token("INTRODUCTION")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': False,
               'any_digit': False,
               'any_punct': False,
               'init_upper': True,
               'init_lower': False,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': False,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': False,
               'typical_upper': True,
               'typical_lower': False,
               'apostrophe_end': False}
        assert all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names)
        
        
        tok = Token("is_1!!_IRR_demo")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': True,
               'any_lower': True,
               'any_digit': True,
               'any_punct': True,
               'init_upper': False,
               'init_lower': True,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': True,
               'any_noninit_digit': True,
               'any_noninit_punct': True,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        assert all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names)
        
        
        tok = Token("0是another@DEMO's")
        ans = {'any_ascii': True,
               'any_non_ascii': True,
               'any_upper': True,
               'any_lower': True,
               'any_digit': True,
               'any_punct': True,
               'init_upper': False,
               'init_lower': False,
               'init_digit': True,
               'init_punct': False,
               'any_noninit_upper': True,
               'any_noninit_lower': True,
               'any_noninit_digit': False,
               'any_noninit_punct': True,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': True}
        assert all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names)
        
        tok = Token("s")
        ans = {'any_ascii': True,
               'any_non_ascii': False,
               'any_upper': False,
               'any_lower': True,
               'any_digit': False,
               'any_punct': False,
               'init_upper': False,
               'init_lower': True,
               'init_digit': False,
               'init_punct': False,
               'any_noninit_upper': False,
               'any_noninit_lower': False,
               'any_noninit_digit': False,
               'any_noninit_punct': False,
               'typical_title': False,
               'typical_upper': False,
               'typical_lower': False,
               'apostrophe_end': False}
        assert all(tok.get_en_shape_feature(key) == ans[key] for key in Token.en_shape_feature_names)
        
        
    def test_numbers(self):
        tok = Token("5.44")
        assert tok.raw_text == "5.44"
        assert tok.text == '<real1>'
        assert tok.get_num_feature('<real1>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<real1>')
        
        tok = Token("-5.44")
        assert tok.raw_text == "-5.44"
        assert tok.text == '<-real1>'
        assert tok.get_num_feature('<-real1>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-real1>')
            
        tok = Token("4")
        assert tok.raw_text == "4"
        assert tok.text == "4"
        assert tok.get_num_feature('<int1>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int1>')
        
        tok = Token("-4")
        assert tok.raw_text == "-4"
        assert tok.text == "-4"
        assert tok.get_num_feature('<-int1>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int1>')
        
        tok = Token("511")
        assert tok.raw_text == "511"
        assert tok.text == "<int3>"
        assert tok.get_num_feature('<int3>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int3>')
        
        tok = Token("-511")
        assert tok.raw_text == "-511"
        assert tok.text == "<-int3>"
        assert tok.get_num_feature('<-int3>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int3>')
        
        tok = Token("2011")
        assert tok.raw_text == "2011"
        assert tok.text == "2011"
        assert tok.get_num_feature('<int4+>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<int4+>')
        
        tok = Token("-2011")
        assert tok.raw_text == "-2011"
        assert tok.text == "<-int4+>"
        assert tok.get_num_feature('<-int4+>')
        assert not any(tok.get_num_feature(mark) for mark in Token.num_feature_names if mark != '<-int4+>')
        
        

class TestTokenSequence(object):
    def test_text(self):
        token_list = [Token(tok) for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        assert tokens.raw_text == ["This", "is", "a", "-3.14", "demo", "."]
        assert tokens.text == ["this", "is", "a", "<-real1>", "demo", "."]
        
    def test_ngrams(self):
        token_list = [Token(tok) for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        assert tokens.bigram == ["this-<sep>-is", "is-<sep>-a", "a-<sep>-<-real1>", 
                                 "<-real1>-<sep>-demo", "demo-<sep>-.", ".-<sep>-<pad>"]
        assert tokens.trigram == ["this-<sep>-is-<sep>-a", "is-<sep>-a-<sep>-<-real1>", 
                                  "a-<sep>-<-real1>-<sep>-demo", "<-real1>-<sep>-demo-<sep>-.", 
                                  "demo-<sep>-.-<sep>-<pad>", ".-<sep>-<pad>-<sep>-<pad>"]
    