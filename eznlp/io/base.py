# -*- coding: utf-8 -*-
from typing import List, Union

from ..token import TokenSequence


class IO(object):
    """An IO interface."""

    def __init__(
        self,
        is_tokenized: bool,
        tokenize_callback=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs
    ):
        self.is_tokenized = is_tokenized
        self.tokenize_callback = tokenize_callback
        assert not self.is_tokenized or self.tokenize_callback is None

        self.encoding = encoding
        self.verbose = verbose
        self.token_kwargs = token_kwargs

    def _build_tokens(self, text: Union[str, List[str]], **kwargs):
        if self.is_tokenized:
            return TokenSequence.from_tokenized_text(
                text, **kwargs, **self.token_kwargs
            )
        else:
            return TokenSequence.from_raw_text(
                text, self.tokenize_callback, **kwargs, **self.token_kwargs
            )

    def read(self, file_path):
        raise NotImplementedError("Not Implemented `read`")
