# -*- coding: utf-8 -*-
import re
import string
from collections import OrderedDict
from functools import cached_property
from typing import Iterable, List

import hanziconv
import jieba
import numpy
import spacy

# "".join([chr(i) for i in range(8211, 8232)])
# "".join([chr(i) for i in range(12289, 12352)])
# "".join([chr(i) for i in range(65091, 65511)])
zh_punctuation = (
    "–—―‖‗‘’‚‛“”„‟†‡•‣․‥…‧"
    + "、。〃〄々〆〇〈〉《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩〪〭〮〯〫〬〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿"
    + "﹃﹄﹅﹆﹇﹈﹉﹊﹋﹌﹍﹎﹏﹐﹑﹒﹔﹕﹖﹗﹘﹙﹚﹛﹜﹝﹞﹟﹠﹡﹢﹣﹤﹥﹦﹨﹩﹪﹫"
    + "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～｟｠｡｢｣､･"
    + "￥￦"
)

# Full-width characters
fw_digits = "０１２３４５６７８９"
fw_uppercase = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
fw_lowercase = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"

assert not any(
    c.isascii() for c in zh_punctuation + fw_digits + fw_uppercase + fw_lowercase
)


ascii_re = re.compile("[\x00-\xff]")
lower_re = re.compile("[a-z]")
upper_re = re.compile("[A-Z]")
digit_re = re.compile("\d")
punct_re = re.compile("[" + "".join("\\" + p for p in string.punctuation) + "]")
non_ascii_re = re.compile("[^\x00-\xff]")

# CJK Unified Ideographs
# https://zh.wikipedia.org/wiki/%E4%B8%AD%E6%97%A5%E9%9F%93%E7%B5%B1%E4%B8%80%E8%A1%A8%E6%84%8F%E6%96%87%E5%AD%97
unihan93_re = re.compile("[\u4e00-\u9fa5〇﨎﨏﨑﨓﨔﨟﨡﨣﨤﨧﨨﨩]")

zh_char_re = re.compile("[\u4e00-\u9fa5]")
zh_punct_re = re.compile("[" + zh_punctuation + "]")
fw_lower_re = re.compile("[" + fw_lowercase + "]")
fw_upper_re = re.compile("[" + fw_uppercase + "]")
fw_digit_re = re.compile("[" + fw_digits + "]")


en_title_word_re = re.compile("[A-Z]{1}[a-z]{1,}")
en_upper_word_re = re.compile("[A-Z]{2,}")
en_lower_word_re = re.compile("[a-z]{2,}")

en_shape2criterion = [
    ("any_ascii", lambda x: ascii_re.search(x) is not None),
    ("any_non_ascii", lambda x: non_ascii_re.search(x) is not None),
    ("any_upper", lambda x: upper_re.search(x) is not None),
    ("any_lower", lambda x: lower_re.search(x) is not None),
    ("any_digit", lambda x: digit_re.search(x) is not None),
    ("any_punct", lambda x: punct_re.search(x) is not None),
    ("init_upper", lambda x: upper_re.search(x[0]) is not None),
    ("init_lower", lambda x: lower_re.search(x[0]) is not None),
    ("init_digit", lambda x: digit_re.search(x[0]) is not None),
    ("init_punct", lambda x: punct_re.search(x[0]) is not None),
    ("any_noninit_upper", lambda x: upper_re.search(x[1:]) is not None),
    ("any_noninit_lower", lambda x: lower_re.search(x[1:]) is not None),
    ("any_noninit_digit", lambda x: digit_re.search(x[1:]) is not None),
    ("any_noninit_punct", lambda x: punct_re.search(x[1:]) is not None),
    ("typical_title", lambda x: en_title_word_re.fullmatch(x) is not None),
    ("typical_upper", lambda x: en_upper_word_re.fullmatch(x) is not None),
    ("typical_lower", lambda x: en_lower_word_re.fullmatch(x) is not None),
    ("apostrophe_end", lambda x: x[-1] == "'" or x[-2:].lower() == "'s"),
]
en_shape2criterion = OrderedDict(en_shape2criterion)

stopwords = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


class Full2Half(object):
    """Translate full-width characters to half-widths"""

    _f2h = {fc: hc for fc, hc in zip(range(0xFF01, 0xFF5E), range(0x21, 0x7E))}
    _f2h.update({0x3000: 0x20})
    _h2f = {hc: fc for fc, hc in _f2h.items()}

    @staticmethod
    def full2half(text):
        return text.translate(Full2Half._f2h)

    @staticmethod
    def half2full(text):
        return text.translate(Full2Half._h2f)


SHORT_LEN = 3


def _adaptive_lower(text: str):
    if len(text) <= 1 or text.islower():
        return text

    lowered = text.lower()
    if lowered in stopwords:
        return lowered

    if len(text) > SHORT_LEN and en_title_word_re.fullmatch(text):
        return lowered

    if len(text) > (SHORT_LEN + 2) and en_upper_word_re.fullmatch(text):
        return lowered

    return text


_case_normalizers = {
    "none": lambda x: x,
    "lower": lambda x: x.lower(),
    "adaptive-lower": lambda x: _adaptive_lower(x),
}


MAX_DIGITS = 4


def _text_to_num_mark(text: str, return_nan_mark: bool = True):
    if text.endswith("%"):
        text4num = text[:-1]
        is_percent = True
    else:
        text4num = text
        is_percent = False

    try:
        possible_value = float(text4num)
    except:
        if return_nan_mark:
            return "<nan>"
        else:
            return text
    else:
        if abs(possible_value) < 1:
            digits = 0
        else:
            digits = min(MAX_DIGITS, int(numpy.log10(abs(possible_value))) + 1)

        if is_percent:
            num_type = "percent"
        elif "." in text4num:
            num_type = "real"
        else:
            num_type = "int"

        if possible_value < 0:
            num_sign = "-"
        else:
            num_sign = ""

        return f"<{num_sign}{num_type}{digits}>"


_number_normalizers = {
    "none": lambda x: x,
    "marks": lambda x: _text_to_num_mark(x, return_nan_mark=False),
    "zeros": lambda x: digit_re.sub("0", x),
}


def _pipeline(*normalizers):
    def pipeline_normalizer(x):
        for f in normalizers:
            x = f(x)
        return x

    return pipeline_normalizer


class Token(object):
    """A token at the modeling level (e.g., word level for English text, or character level for Chinese text).

    `Token` provides access to lower-level attributes (prefixes, suffixes).
    """

    _en_shape_feature_names = list(en_shape2criterion.keys())

    _basic_ohot_fields = [
        "text",
        "num_mark",
        "prefix_2",
        "prefix_3",
        "prefix_4",
        "prefix_5",
        "suffix_2",
        "suffix_3",
        "suffix_4",
        "suffix_5",
        "en_pattern",
        "en_pattern_sum",
    ]
    _basic_mhot_fields = ["en_shape_features"]

    def __init__(
        self,
        raw_text: str,
        pre_text_normalizer=None,
        case_mode="None",
        number_mode="None",
        to_half=True,
        to_zh_simplified=False,
        post_text_normalizer=None,
        **kwargs,
    ):
        self.raw_text = raw_text
        if callable(pre_text_normalizer):
            self.raw_text = pre_text_normalizer(self.raw_text)

        pipeline_normalizer = _pipeline(
            _case_normalizers[case_mode.lower()],
            _number_normalizers[number_mode.lower()],
            lambda x: Full2Half.full2half(x) if to_half else x,
            lambda x: hanziconv.HanziConv.toSimplified(x) if to_zh_simplified else x,
        )
        self.text = pipeline_normalizer(self.raw_text)
        if callable(post_text_normalizer):
            self.text = post_text_normalizer(self.text)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        return (
            isinstance(other, Token)
            and self.raw_text == other.raw_text
            and self.text == other.text
        )

    def __len__(self):
        return len(self.raw_text)

    def __repr__(self):
        return self.raw_text

    @property
    def prefix_2(self):
        return self.raw_text[:2]

    @property
    def prefix_3(self):
        return self.raw_text[:3]

    @property
    def prefix_4(self):
        return self.raw_text[:4]

    @property
    def prefix_5(self):
        return self.raw_text[:5]

    @property
    def suffix_2(self):
        return self.raw_text[-2:]

    @property
    def suffix_3(self):
        return self.raw_text[-3:]

    @property
    def suffix_4(self):
        return self.raw_text[-4:]

    @property
    def suffix_5(self):
        return self.raw_text[-5:]

    @property
    def num_mark(self):
        return _text_to_num_mark(self.raw_text, return_nan_mark=True)

    @property
    def en_pattern(self):
        feature = upper_re.sub("A", self.raw_text)
        feature = lower_re.sub("a", feature)
        feature = digit_re.sub("0", feature)
        return feature

    @property
    def en_pattern_sum(self):
        feature = self.en_pattern
        feature = re.sub("A+", "A", feature)
        feature = re.sub("a+", "a", feature)
        feature = re.sub("0+", "0", feature)
        return feature

    @property
    def en_shape_features(self):
        return numpy.array(
            [criterion(self.raw_text) for criterion in en_shape2criterion.values()]
        )

    @property
    def zh_shape_features(self):
        return None


class TokenSequence(object):
    """A wrapper of token list, providing sequential attribute access to all tokens."""

    _softword_idx2tag = ["B", "M", "E", "S"]
    _softword_tag2idx = {t: i for i, t in enumerate(_softword_idx2tag)}

    def __init__(
        self,
        token_list: List[Token],
        token_sep=" ",
        pad_token="<pad>",
        none_token="<none>",
    ):
        self.token_list = token_list
        self.token_sep = token_sep
        self.pad_token = pad_token
        self.none_token = none_token
        assert len(self.token_sep) <= 1

    def __getattr__(self, name):
        # NOTE: `__attr__` method is only invoked if the attribute wasn't found the usual ways, so
        # it is good for implementing a fallback for missing attributes. While, `__getattribute__`
        # is invoked before looking at the actual attributes on the object.
        # See: https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
        if len(self.token_list) == 0:
            # Unable to check attribute existence, return an empty list anyway
            return []
        elif hasattr(self.token_list[0], name):
            return [getattr(tok, name) for tok in self.token_list]
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def __eq__(self, other):
        return (
            isinstance(other, TokenSequence)
            and self.__getstate__() == other.__getstate__()
        )

    def __len__(self):
        return len(self.token_list)

    def __repr__(self):
        return repr(self.token_list)

    @property
    def _tokens_kwargs(self):
        return {
            "token_sep": self.token_sep,
            "pad_token": self.pad_token,
            "none_token": self.none_token,
        }

    def __getstate__(self):
        return {"token_list": self.token_list, **self._tokens_kwargs}

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.token_list[i]
        elif isinstance(i, slice):
            return TokenSequence(self.token_list[i], **self._tokens_kwargs)
        else:
            raise TypeError(f"Invalid subscript type of {i}")

    def __add__(self, other):
        assert isinstance(other, TokenSequence)
        assert other._tokens_kwargs == self._tokens_kwargs
        return TokenSequence(self.token_list + other.token_list, **self._tokens_kwargs)

    def build_pseudo_boundaries(self, sep_width: int = None):
        if sep_width is None:
            sep_width = len(self.token_sep)

        token_lens = [len(tok) for tok in self.token_list]
        self.start = [0] + numpy.cumsum(numpy.array(token_lens) + sep_width).tolist()[
            :-1
        ]
        self.end = [s + l for s, l in zip(self.start, token_lens)]

    def _assert_for_softwords(self, tokenize_callback):
        assert self.token_sep == ""
        assert hasattr(tokenize_callback, "__self__")
        assert isinstance(
            tokenize_callback.__self__, (jieba.Tokenizer, LexiconTokenizer)
        )
        assert tokenize_callback.__name__.startswith("tokenize")

    def build_softwords(self, tokenize_callback, **kwargs):
        self._assert_for_softwords(tokenize_callback)

        self.softword = [
            numpy.zeros(len(self._softword_idx2tag), dtype=bool)
            for tok in self.token_list
        ]

        for word_text, word_start, word_end in tokenize_callback(
            self.token_sep.join(self.raw_text), **kwargs
        ):
            if word_end - word_start == 1:
                self.softword[word_start][self._softword_tag2idx["S"]] = True
            else:
                self.softword[word_start][self._softword_tag2idx["B"]] = True
                self.softword[word_end - 1][self._softword_tag2idx["E"]] = True
                for k in range(word_start + 1, word_end - 1):
                    self.softword[k][self._softword_tag2idx["M"]] = True

    def build_softlexicons(self, tokenize_callback, **kwargs):
        self._assert_for_softwords(tokenize_callback)

        self.softlexicon = [
            [[] for t in self._softword_idx2tag] for tok in self.token_list
        ]

        for word_text, word_start, word_end in tokenize_callback(
            self.token_sep.join(self.raw_text), **kwargs
        ):
            if word_end - word_start == 1:
                self.softlexicon[word_start][self._softword_tag2idx["S"]].append(
                    word_text
                )
            else:
                self.softlexicon[word_start][self._softword_tag2idx["B"]].append(
                    word_text
                )
                self.softlexicon[word_end - 1][self._softword_tag2idx["E"]].append(
                    word_text
                )
                for k in range(word_start + 1, word_end - 1):
                    self.softlexicon[k][self._softword_tag2idx["M"]].append(word_text)

        # Add a special token to empty word sets
        for word_sets in self.softlexicon:
            for word_set in word_sets:
                if len(word_set) == 0:
                    word_set.append(self.none_token)

    @cached_property
    def bigram(self):
        unigram = self.text
        return [
            self.token_sep.join(gram)
            for gram in zip(unigram, unigram[1:] + [self.pad_token])
        ]

    @cached_property
    def trigram(self):
        unigram = self.text
        return [
            self.token_sep.join(gram)
            for gram in zip(
                unigram,
                unigram[1:] + [self.pad_token],
                unigram[2:] + [self.pad_token, self.pad_token],
            )
        ]

    def spans_within_max_length(self, max_len: int):
        total_len = len(self.token_list)
        slice_start = 0

        while True:
            if total_len - slice_start <= max_len:
                yield slice(slice_start, total_len)
                break
            else:
                slice_end = slice_start + max_len
                while not self.token_list[slice_end - 1].text in (".", "?", "!", ";"):
                    slice_end -= 1
                    if slice_end <= slice_start:
                        raise ValueError(
                            f"Cannot find proper slices in {self.token_list[slice_start:slice_start+max_len]}"
                        )
                yield slice(slice_start, slice_end)
                slice_start = slice_end

    def attach_additional_tags(
        self, additional_tags: dict = None, additional_tok2tags: list = None
    ):
        """

        Parameters
        ----------
        additional_tags : dict of lists, optional
            {tag_name: tags, ...}.
        additional_tok2tags : list of tuples, optional
            [(tag_name: str, tok2tag: dict), ...].
        """
        if additional_tags is not None:
            for tag_name, tags in additional_tags.items():
                for tok, tag in zip(self.token_list, tags):
                    setattr(tok, tag_name, tag)

        if additional_tok2tags is not None:
            for tag_name, tok2tag in additional_tok2tags:
                for tok in self.token_list:
                    setattr(tok, tag_name, tok2tag.get(tok.text, tok2tag["<unk>"]))

        return self

    @classmethod
    def from_tokenized_text(
        cls,
        tokenized_text: List[str],
        additional_tags=None,
        additional_tok2tags=None,
        token_sep=" ",
        pad_token="<pad>",
        none_token="<none>",
        **kwargs,
    ):
        """Build `TokenSequence` from tokenized text.

        Parameters
        ----------
        tokenized_text: List[str]
            A list of tokenized text.
        """
        token_lens = [len(tok) for tok in tokenized_text]
        token_starts = [0] + numpy.cumsum(
            numpy.array(token_lens) + len(token_sep)
        ).tolist()[:-1]
        token_ends = [s + l for s, l in zip(token_starts, token_lens)]

        token_list = [
            Token(tok_text, start=s, end=e, **kwargs)
            for tok_text, s, e in zip(tokenized_text, token_starts, token_ends)
        ]
        tokens = cls(
            token_list, token_sep=token_sep, pad_token=pad_token, none_token=none_token
        )
        tokens.attach_additional_tags(
            additional_tags=additional_tags, additional_tok2tags=additional_tok2tags
        )
        return tokens

    @classmethod
    def from_raw_text(
        cls,
        raw_text: str,
        tokenize_callback=None,
        additional_tok2tags=None,
        token_sep=" ",
        pad_token="<pad>",
        none_token="<none>",
        **kwargs,
    ):
        """Build `TokenSequence` from raw text.

        Parameters
        ----------
        raw_text: str
            A string of raw text.
        tokenize_callback: `None`, str or callable
            (1) `None`, "space": split text by space.
            (2) "char": split text into characters.
            (3) spacy.language.Language, jieba.Tokenizer.cut, jieba.Tokenizer.tokenize: split text by given tokenize method.
        """
        if tokenize_callback is None or (
            isinstance(tokenize_callback, str)
            and tokenize_callback.lower().startswith("space")
        ):
            # token_list = [Token(tok_text, **kwargs) for tok_text in raw_text.split()]
            space_spans = [space.span() for space in re.finditer("\s+", raw_text)]
            token_spans = [
                (s, e)
                for s, e in zip(
                    [0] + [s[1] for s in space_spans],
                    [s[0] for s in space_spans] + [len(raw_text)],
                )
                if s < e
            ]
            token_list = [
                Token(raw_text[s:e], start=s, end=e, **kwargs) for s, e in token_spans
            ]
        elif isinstance(
            tokenize_callback, str
        ) and tokenize_callback.lower().startswith("char"):
            token_list = [
                Token(tok_text, start=k, end=k + 1, **kwargs)
                for k, tok_text in enumerate(raw_text)
            ]
        elif isinstance(tokenize_callback, spacy.language.Language):
            token_list = [
                Token(tok.text, start=tok.idx, end=tok.idx + len(tok.text), **kwargs)
                for tok in tokenize_callback(raw_text)
            ]
        elif hasattr(tokenize_callback, "__self__") and isinstance(
            tokenize_callback.__self__, jieba.Tokenizer
        ):
            if tokenize_callback.__name__.startswith("tokenize"):
                token_list = [
                    Token(tok_text, start=tok_start, end=tok_end, **kwargs)
                    for tok_text, tok_start, tok_end in tokenize_callback(raw_text)
                ]
            elif tokenize_callback.__name__.startswith("cut"):
                token_list = [
                    Token(tok_text, **kwargs)
                    for tok_text in tokenize_callback(raw_text)
                ]
            else:
                raise ValueError(
                    f"Invalid method of `jieba.Tokenizer`: {tokenize_callback}"
                )
        else:
            raise ValueError(f"Invalid `tokenize_callback`: {tokenize_callback}")

        tokens = cls(
            token_list, token_sep=token_sep, pad_token=pad_token, none_token=none_token
        )
        tokens.attach_additional_tags(additional_tok2tags=additional_tok2tags)
        return tokens

    def to_raw_text(self):
        """Convert `TokenSequence` to raw text."""
        if hasattr(self, "start") and hasattr(self, "end"):
            token_sep = " " if len(self.token_sep) == 0 else self.token_sep
            spaces = [
                token_sep * (e - s) for s, e in zip([0] + self.end[:-1], self.start)
            ]
            return "".join(
                t for space_token in zip(spaces, self.raw_text) for t in space_token
            )
        else:
            return self.token_sep.join(self.raw_text)


class LexiconTokenizer(object):
    def __init__(
        self, lexicon: Iterable[str], max_len: int = 10, return_singleton: bool = False
    ):
        self.lexicon = set(lexicon)
        self.max_len = max_len
        self.return_singleton = return_singleton

    def tokenize(self, text: str):
        L = len(text)
        for word_start in range(L):
            for word_end in range(
                word_start + 1, min(word_start + self.max_len, L) + 1
            ):
                word_text = text[word_start:word_end]
                if (self.return_singleton and word_end - word_start == 1) or (
                    word_text in self.lexicon
                ):
                    yield (word_text, word_start, word_end)


def custom_spacy_tokenizer(
    nlp, custom_prefixes=None, custom_suffixes=None, custom_infixes=None
):
    """
    References:
    https://spacy.io/usage/linguistic-features#tokenization
    http://www.longest.io/2018/01/27/spacy-custom-tokenization.html
    """
    if custom_prefixes is None:
        prefix_search = nlp.tokenizer.prefix_search
    else:
        prefix_search = spacy.util.compile_prefix_regex(
            tuple(list(nlp.Defaults.prefixes) + custom_prefixes)
        ).search
    if custom_suffixes is None:
        suffix_search = nlp.tokenizer.suffix_search
    else:
        suffix_search = spacy.util.compile_suffix_regex(
            tuple(list(nlp.Defaults.suffixes) + custom_suffixes)
        ).search
    if custom_infixes is None:
        infix_finditer = nlp.tokenizer.infix_finditer
    else:
        infix_finditer = spacy.util.compile_infix_regex(
            tuple(list(nlp.Defaults.infixes) + custom_infixes)
        ).finditer

    return spacy.tokenizer.Tokenizer(
        nlp.vocab,
        rules=nlp.tokenizer.rules,
        prefix_search=prefix_search,
        infix_finditer=infix_finditer,
        suffix_search=suffix_search,
        token_match=nlp.tokenizer.token_match,
        url_match=nlp.tokenizer.url_match,
    )
