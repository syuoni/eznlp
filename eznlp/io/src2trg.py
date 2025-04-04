# -*- coding: utf-8 -*-
import re
from typing import List, Union

import tqdm

from ..token import TokenSequence
from .base import IO


class Src2TrgIO(IO):
    """An IO interface of source-to-target files."""

    def __init__(
        self,
        tokenize_callback=None,
        trg_tokenize_callback=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs
    ):
        super().__init__(
            is_tokenized=False,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs
        )
        self.trg_tokenize_callback = trg_tokenize_callback

    def _build_trg_tokens(self, text: Union[str, List[str]], **kwargs):
        if self.is_tokenized:
            return TokenSequence.from_tokenized_text(
                text, **kwargs, **self.token_kwargs
            )
        else:
            return TokenSequence.from_raw_text(
                text, self.trg_tokenize_callback, **kwargs, **self.token_kwargs
            )

    def read(self, src_path, trg_path):
        data = []
        with open(src_path, "r", encoding=self.encoding) as f:
            # Replace consecutive spaces with a single space
            src_lines = [
                re.sub("\s+", " ", line.strip()) for line in f if line.strip() != ""
            ]
        with open(trg_path, "r", encoding=self.encoding) as f:
            trg_lines = [
                re.sub("\s+", " ", line.strip()) for line in f if line.strip() != ""
            ]

        assert len(src_lines) == len(trg_lines)
        for src_line, trg_line in tqdm.tqdm(
            zip(src_lines, trg_lines),
            total=len(src_lines),
            disable=not self.verbose,
            ncols=100,
            desc="Loading src2trg data",
        ):
            src_tokens = self._build_tokens(src_line)
            trg_tokens = self._build_trg_tokens(trg_line)
            data.append({"tokens": src_tokens, "full_trg_tokens": [trg_tokens]})
        return data
