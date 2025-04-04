# -*- coding: utf-8 -*-
import glob
from typing import List

import tqdm

from .base import IO


class CategoryFolderIO(IO):
    """An IO interface of text files in category-specific folders."""

    def __init__(
        self,
        categories: List[str],
        tokenize_callback=None,
        mapping=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        self.categories = categories
        self.mapping = {} if mapping is None else mapping
        super().__init__(
            is_tokenized=False,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )

    def read(self, folder_path):
        data = []
        for label in self.categories:
            file_paths = glob.glob(f"{folder_path}/{label}/*.txt")
            for path in tqdm.tqdm(
                file_paths,
                disable=not self.verbose,
                ncols=100,
                desc=f"Loading data in folder {label}",
            ):
                with open(path, encoding=self.encoding) as f:
                    raw_text = f.read()
                for pattern, repl in self.mapping.items():
                    raw_text = raw_text.replace(pattern, repl)

                tokens = self._build_tokens(raw_text)
                data.append({"tokens": tokens, "label": label})

        return data
