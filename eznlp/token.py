# -*- coding: utf-8 -*-
import string
import re
from collections import OrderedDict
from hanziconv import HanziConv
import spacy
from spacy.tokenizer import Tokenizer
import numpy as np

zh_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－——／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"

num_mark2re = [('<int1>', '\d{1}'), 
               ('<int2>', '\d{2}'), 
               ('<int3>', '\d{3}'), 
               ('<int4+>', '\d{4,}'), 
               ('<real0>', '0\.\d*'), 
               ('<real1>', '[1-9]\.\d*'), 
               ('<real2>', '\d{2}\.\d*'), 
               ('<real3>', '\d{3}\.\d*'), 
               ('<real4+>', '\d{4,}\.\d*')]
num_mark2re = OrderedDict([(mark, re.compile(re_expr)) for mark, re_expr in num_mark2re])
preserve_nums = set(range(-10, 11)) | set(range(1900, 2101))

ascii_re = re.compile('[\x00-\xff]')
lower_re = re.compile('[a-z]')
upper_re = re.compile('[A-Z]')
digit_re = re.compile('\d')
punct_re = re.compile('[' + ''.join("\\" + p for p in string.punctuation) + ']')
non_ascii_re = re.compile('[^\x00-\xff]')
zh_punct_re = re.compile('[' + zh_punctuation + ']')

en_title_word_re = re.compile('[A-Z]{1}[a-z]{1,}')
en_upper_word_re = re.compile('[A-Z]{2,}')
en_lower_word_re = re.compile('[a-z]{2,}')

en_shape2criterion = [('any_ascii', lambda x: ascii_re.search(x) is not None), 
                      ('any_non_ascii', lambda x: non_ascii_re.search(x) is not None), 
                      ('any_upper', lambda x: upper_re.search(x) is not None), 
                      ('any_lower', lambda x: lower_re.search(x) is not None), 
                      ('any_digit', lambda x: digit_re.search(x) is not None), 
                      ('any_punct', lambda x: punct_re.search(x) is not None), 
                      ('init_upper', lambda x: upper_re.search(x[0]) is not None), 
                      ('init_lower', lambda x: lower_re.search(x[0]) is not None), 
                      ('init_digit', lambda x: digit_re.search(x[0]) is not None), 
                      ('init_punct', lambda x: punct_re.search(x[0]) is not None), 
                      ('any_noninit_upper', lambda x: upper_re.search(x[1:]) is not None), 
                      ('any_noninit_lower', lambda x: lower_re.search(x[1:]) is not None), 
                      ('any_noninit_digit', lambda x: digit_re.search(x[1:]) is not None), 
                      ('any_noninit_punct', lambda x: punct_re.search(x[1:]) is not None), 
                      ('typical_title', lambda x: en_title_word_re.fullmatch(x) is not None), 
                      ('typical_upper', lambda x: en_upper_word_re.fullmatch(x) is not None), 
                      ('typical_lower', lambda x: en_lower_word_re.fullmatch(x) is not None), 
                      ('apostrophe_end', lambda x: x[-1] == "'" or x[-2:].lower() == "'s")]
en_shape2criterion = OrderedDict(en_shape2criterion)

stopwords = {"a", "about", "above", "after", "again", "against", "all", "am", 
             "an", "and", "any", "are", "aren't", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", 
             "by", "can't", "cannot", "could", "couldn't", "did", "didn't", 
             "do", "does", "doesn't", "doing", "don't", "down", "during", 
             "each", "few", "for", "from", "further", "had", "hadn't", "has", 
             "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", 
             "he's", "her", "here", "here's", "hers", "herself", "him", 
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
             "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", 
             "itself", "let's", "me", "more", "most", "mustn't", "my", 
             "myself", "no", "nor", "not", "of", "off", "on", "once", "only", 
             "or", "other", "ought", "our", "ours", "ourselves", "out", "over", 
             "own", "same", "shan't", "she", "she'd", "she'll", "she's", 
             "should", "shouldn't", "so", "some", "such", "than", "that", 
             "that's", "the", "their", "theirs", "them", "themselves", "then", 
             "there", "there's", "these", "they", "they'd", "they'll", 
             "they're", "they've", "this", "those", "through", "to", "too", 
             "under", "until", "up", "very", "was", "wasn't", "we", "we'd", 
             "we'll", "we're", "we've", "were", "weren't", "what", "what's", 
             "when", "when's", "where", "where's", "which", "while", "who", 
             "who's", "whom", "why", "why's", "with", "won't", "would", 
             "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
             "yours", "yourself", "yourselves"}
SHORT_LEN = 3

class Full2Half(object):
    '''Translate full-width characters to half-widths
    '''
    _f2h = {fc: hc for fc, hc in zip(range(0xFF01, 0xFF5E), range(0x21, 0x7E))}
    _f2h.update({0x3000: 0x20})
    _h2f = {hc: fc for fc, hc in _f2h.items()}
    
    @staticmethod
    def full2half(text):
        return text.translate(Full2Half._f2h)
    
    @staticmethod
    def half2full(text):
        return text.translate(Full2Half._h2f)
    
    
def adaptive_lower(text):
    if len(text) <= 1 or text.islower():
        return text
    
    lowered = text.lower()
    if lowered in stopwords:
        return lowered
    
    if len(text) > SHORT_LEN and en_title_word_re.fullmatch(text):
        return lowered
    
    if len(text) > (SHORT_LEN+2) and en_upper_word_re.fullmatch(text):
        return lowered
    
    return text
    

class Token(object):
    en_shape_feature_names = list(en_shape2criterion.keys())
    num_feature_names = list(num_mark2re.keys()) + [mark[0] + '-' + mark[1:] for mark in num_mark2re.keys()]
    
    def __init__(self, raw_text, to_lower='adaptive_lower', to_half=True, to_zh_simplified=False, 
                 to_num_marks=True, **kwargs):
        self.raw_text = raw_text
        if to_lower == 'adaptive_lower':
            self.text = adaptive_lower(raw_text)
        elif to_lower == 'all_lower':
            self.text = raw_text.lower()
        elif to_lower == 'not_lower':
            self.text = raw_text
        else:
            raise ValueError(f"Invalid value of to_lower parameter: {to_lower}")
            
        self.text = Full2Half.full2half(self.text) if to_half else self.text
        self.text = HanziConv.toSimplified(self.text) if to_zh_simplified else self.text
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self._build_en_pattern_features()
        self._build_prefix_features()
        self._build_suffix_features()
        self._build_en_shape_features()
        self._build_num_features(to_num_marks)
        
    def _build_prefix_features(self, min_win=2, max_win=5):
        """
        Prefix features, built on text.
        """
        for win in range(min_win, max_win+1):
            setattr(self, f'prefix_{win}', self.text[:win])
            
    def _build_suffix_features(self, min_win=2, max_win=5):
        """
        Suffix features, built on text. 
        """
        for win in range(min_win, max_win+1):
            setattr(self, f'suffix_{win}', self.text[-win:])
        
    def _build_num_features(self, to_num_marks=True):
        """
        Number features, built on text.
        """
        features = [False] * (2 * len(num_mark2re))
        
        if digit_re.search(self.text) is not None:
            if self.text.startswith('-'):
                text4match = self.text[1:]
                negative = True
            elif self.text.startswith('+'):
                text4match = self.text[1:]
                negative = False
            else:
                text4match = self.text
                negative = False
                
            for k, (mark, num_re) in enumerate(num_mark2re.items()):
                if num_re.fullmatch(text4match) is not None:
                    offset = len(num_mark2re) if negative else 0
                    features[k + offset] = True
                    if float(self.text) not in preserve_nums:
                        self.text = mark[0] + '-' + mark[1:] if negative else mark
                    break
                
        self.num_features = features
        
    def get_num_feature(self, key):
        return self.num_features[Token.num_feature_names.index(key)]
        
    def _build_en_pattern_features(self):
        """
        English pattern features, built on raw_text.
        """
        feature = upper_re.sub('A', self.raw_text)
        feature = lower_re.sub('a', feature)
        feature = digit_re.sub('0', feature)
        self.en_pattern = feature
        
        feature = re.sub('A+', 'A', feature)
        feature = re.sub('a+', 'a', feature)
        feature = re.sub('0+', '0', feature)
        self.en_pattern_sum = feature
        
    def _build_en_shape_features(self):
        """
        English word shape features, built on raw_text.
        """
        features = [criterion(self.raw_text) for criterion in en_shape2criterion.values()]
        self.en_shape_features = features
        
    def get_en_shape_feature(self, key):
        return self.en_shape_features[Token.en_shape_feature_names.index(key)]
        
    def _build_zh_features(self):
        pass
        
    def __str__(self):
        return self.raw_text
    
    def __repr__(self):
        return self.raw_text
    
    
class TokenSequence(object):
    def __init__(self, token_list, max_len=None):
        self.token_list = token_list
        
    def __getattr__(self, name):
        # NOTE: `__attr__` method is only invoked if the attribute wasn't found the usual ways, so 
        # it is good for implementing a fallback for missing attributes. While, `__getattribute__`
        # is invoked before looking at the actual attributes on the object. 
        # See: https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
        if hasattr(self.token_list[0], name):
            return [getattr(tok, name) for tok in self.token_list]
        else:
            raise AttributeError(f"{type(self)} object has no attribute {name}")
    
    @property
    def bigram(self):
        unigram = self.text
        return ['-<sep>-'.join(gram) for gram in zip(unigram, unigram[1:] + ['<pad>'])]
    
    @property
    def trigram(self):
        unigram = self.text
        return ['-<sep>-'.join(gram) for gram in zip(unigram, unigram[1:] + ['<pad>'], unigram[2:] + ['<pad>', '<pad>'])]
    
    def __getstate__(self):
        return self.token_list

    def __setstate__(self, token_list):
        self.token_list = token_list
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.token_list[i]
        elif isinstance(i, slice):
            return TokenSequence(self.token_list[i])
        else:
            raise TypeError(f"Invalid subscript type of {i}")
            
    def build_word_pieces(self, tokenizer, rebuild=False):
        if not hasattr(self, 'word_pieces') or rebuild:
            nested_word_pieces = [tokenizer.tokenize(word) for word in self.raw_text]
            self.word_pieces = [sub_word for i, word in enumerate(nested_word_pieces) for sub_word in word]
            self.word_piece_tok_pos = [i for i, word in enumerate(nested_word_pieces) for sub_word in word]
            # self.tok_word_piece_pos = [0] + np.cumsum([len(word) for word in nested_word_pieces[:-1]]).tolist()
        
        
    def spans_within_max_length(self, max_len):
        total_len = len(self.token_list)
        slice_start = 0
        
        while True:
            if total_len - slice_start <= max_len:
                yield slice(slice_start, total_len)
                break
            else:    
                slice_end = slice_start + max_len
                while not self.token_list[slice_end-1].text in ('.', '?', '!', ';'):
                    slice_end -= 1
                    if slice_end <= slice_start:
                        raise ValueError(f"Cannot find proper slices in {self.token_list[slice_start:slice_start+max_len]}")
                yield slice(slice_start, slice_end)
                slice_start = slice_end
        
    
    def __len__(self):
        return len(self.token_list)
    
    def __str__(self):
        return str(self.token_list)
    
    def __repr__(self):
        return repr(self.token_list)
    
    
    @classmethod
    def from_tokenized_text(cls, tokenized_text: list, **kwargs):
        token_list = [Token(tok_text, **kwargs) for tok_text in tokenized_text]
        return cls(token_list)
    
    
    @classmethod
    def from_raw_text(cls, raw_text: str, spacy_nlp, additional_tok2tags=None, **kwargs):
        """
        `additional_tok2tags`: [(tag_name: str, tok2tag: dict), ...]
        """
        token_list = [Token(tok.text, start=tok.idx, end=tok.idx+len(tok.text), lemma=tok.lemma_, 
                            upos=tok.pos_, detailed_pos=tok.tag_, ent_tag='-'.join([tok.ent_iob_, tok.ent_type_]), 
                            dep=tok.dep_, **kwargs) for tok in spacy_nlp(raw_text)]
        
        if additional_tok2tags is not None:
            for tag_name, tok2tag in additional_tok2tags:
                for tok in token_list:
                    setattr(tok, tag_name, tok2tag.get(tok.text, tok2tag['<unk>']))
        
        return cls(token_list)


def custom_spacy_tokenizer(nlp, custom_prefixes=None, custom_suffixes=None, custom_infixes=None):
    """
    References:
    https://spacy.io/usage/linguistic-features#tokenization
    http://www.longest.io/2018/01/27/spacy-custom-tokenization.html
    """
    if custom_prefixes is None:
        prefix_search = nlp.tokenizer.prefix_search
    else:
        prefix_search = spacy.util.compile_prefix_regex(tuple(list(nlp.Defaults.prefixes) + custom_prefixes)).search
    if custom_suffixes is None:
        suffix_search = nlp.tokenizer.suffix_search
    else:
        suffix_search = spacy.util.compile_suffix_regex(tuple(list(nlp.Defaults.suffixes) + custom_suffixes)).search
    if custom_infixes is None:
        infix_finditer = nlp.tokenizer.infix_finditer
    else:
        infix_finditer = spacy.util.compile_infix_regex(tuple(list(nlp.Defaults.infixes) + custom_infixes)).finditer

    return Tokenizer(nlp.vocab, rules=nlp.tokenizer.rules,
                     prefix_search=prefix_search, 
                     infix_finditer=infix_finditer, 
                     suffix_search=suffix_search,
                     token_match=nlp.tokenizer.token_match, 
                     url_match=nlp.tokenizer.url_match) 

    
    