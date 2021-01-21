# -*- coding: utf-8 -*-
import pytest

from eznlp.data.token import Full2Half
from eznlp.data import Token, TokenSequence


def test_full2half():
    assert Full2Half.full2half("，；：？！") == ",;:?!"
    assert Full2Half.half2full(",;:?!") == "，；：？！"
    
    
class TestToken(object):
    def test_assign_attr(self):
        tok = Token("-5.44", chunking='B-NP')
        assert hasattr(tok, 'chunking')
        assert tok.chunking == 'B-NP'
        
    
    @pytest.mark.parametrize("raw_text, lowered_text, expected_en_pattern, expected_en_pattern_sum", 
                             [("Of", "of", "Aa", "Aa"), 
                              ("THE", "the", "AAA", "A"), 
                              ("marry", "marry", "aaaaa", "a"), 
                              ("MARRY", "MARRY", "AAAAA", "A"), 
                              ("WATERMELON", "watermelon", "AAAAAAAAAA", "A"), 
                              ("WATERMELON-", "WATERMELON-", "AAAAAAAAAA-", "A-"), 
                              ("Jack", "jack", "Aaaa", "Aa"), 
                              ("jACK", "jACK", "aAAA", "aA"), 
                              ("1Bc23dEJ", "1Bc23dEJ", "0Aa00aAA", "0Aa0aA")])
    def test_token_basics(self, raw_text, lowered_text, expected_en_pattern, expected_en_pattern_sum):
        tok = Token(raw_text, case_mode='Adaptive-Lower')
        assert tok.text == lowered_text
        
        assert tok.prefix_2 == raw_text[:2]
        assert tok.prefix_3 == raw_text[:3]
        assert tok.prefix_4 == raw_text[:4]
        assert tok.prefix_5 == raw_text[:5]
        assert tok.suffix_2 == raw_text[-2:]
        assert tok.suffix_3 == raw_text[-3:]
        assert tok.suffix_4 == raw_text[-4:]
        assert tok.suffix_5 == raw_text[-5:]
        
        
    @pytest.mark.parametrize("raw_text, expected_en_shape_features", 
                             [("s", {'any_ascii': True,
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
                                     'apostrophe_end': False}), 
                              ("Jack", {'any_ascii': True,
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
                                        'apostrophe_end': False}), 
                              ("INTRODUCTION", {'any_ascii': True,
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
                                                'apostrophe_end': False}), 
                              ("is_1!!_IRR_demo", {'any_ascii': True,
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
                                                   'apostrophe_end': False}), 
                              ("0是another@DEMO's", {'any_ascii': True,
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
                                                     'apostrophe_end': True})])
    def test_en_shape_features(self, raw_text, expected_en_shape_features):
        tok = Token(raw_text)
        for i, key in enumerate(tok.en_shape_feature_names):
            assert tok.en_shape_features[i] == expected_en_shape_features[key]
        
        
    @pytest.mark.parametrize("raw_text, num_mark", 
                             [("5.44", "<real1>"), 
                              ("-5.44", "<-real1>"), 
                              ("4", "<int1>"), 
                              ("-4", "<-int1>"), 
                              ("511", "<int3>"), 
                              ("-511", "<-int3>"), 
                              ("2011", "<int4>"), 
                              ("-2011", "<-int4>"), 
                              ("123456", "<int4>"), 
                              ("-123456", "<-int4>")])
    def test_number_to_mark(self, raw_text, num_mark):
        tok = Token(raw_text, number_mode='Marks')
        assert tok.raw_text == raw_text
        assert tok.text == num_mark
        for i, key in enumerate(tok.num_feature_names):
            assert tok.num_features[i] == (num_mark == key)
            
    
    @pytest.mark.parametrize("raw_text, zeros_text", 
                             [("5.44", "0.00"), 
                              ("-5.44", "-0.00"), 
                              ("Jack888_7John", "Jack000_0John")])
    def test_number_to_zero(self, raw_text, zeros_text):
        tok = Token(raw_text, number_mode='Zeros')
        assert tok.raw_text == raw_text
        assert tok.text == zeros_text
        
        

class TestTokenSequence(object):
    def test_text(self):
        token_list = [Token(tok, case_mode='Lower', number_mode='Marks') for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        assert tokens.raw_text == ["This", "is", "a", "-3.14", "demo", "."]
        assert tokens.text == ["this", "is", "a", "<-real1>", "demo", "."]
        
        tokens.build_pseudo_boundaries(sep_width=1)
        assert [e-s for s, e in zip(tokens.start, tokens.end)] == [len(tok) for tok in tokens.token_list]
        
        
    def test_ngrams(self):
        token_list = [Token(tok, case_mode='Lower', number_mode='Marks') for tok in "This is a -3.14 demo .".split()]
        tokens = TokenSequence(token_list)
        
        assert tokens.bigram == ["this-<sep>-is", "is-<sep>-a", "a-<sep>-<-real1>", 
                                 "<-real1>-<sep>-demo", "demo-<sep>-.", ".-<sep>-<pad>"]
        assert tokens.trigram == ["this-<sep>-is-<sep>-a", "is-<sep>-a-<sep>-<-real1>", 
                                  "a-<sep>-<-real1>-<sep>-demo", "<-real1>-<sep>-demo-<sep>-.", 
                                  "demo-<sep>-.-<sep>-<pad>", ".-<sep>-<pad>-<sep>-<pad>"]
        
        