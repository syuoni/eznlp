# -*- coding: utf-8 -*-
import numpy

from ..utils import ChunksTagsTranslator
from .base import IO


class ConllIO(IO):
    """An IO interface of CoNLL-format files.

    This reader will *break* the proceeding sequence and *skip* the current line starting with any of `sentence_sep_starts` and `document_sep_starts`.
    Hence, if you aim to ignore certain lines, you can place the corresponding "starting markers", if exist, in `sentence_sep_starts`.

    Parameters
    ----------
    sentence_sep_starts: List[str]
        * For OntoNotes (i.e., Conll2011, Conll2012), `sentence_sep_starts` should be `["#end", "pt/"]`
    document_sep_starts: List[str]
        * For Conll2003, `document_sep_starts` should be `["-DOCSTART-"]`
        * For OntoNotes (i.e., Conll2011, Conll2012), `document_sep_starts` should be `["#begin"]`
    """

    def __init__(
        self,
        text_col_id=0,
        tag_col_id=1,
        sep=None,
        scheme="BIO1",
        tag_sep="-",
        breaking_for_types=True,
        additional_col_id2name=None,
        sentence_sep_starts=None,
        document_sep_starts=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs
    ):
        self.text_col_id = text_col_id
        self.tag_col_id = tag_col_id
        self.sep = sep

        self.tags_translator = ChunksTagsTranslator(
            scheme=scheme, sep=tag_sep, breaking_for_types=breaking_for_types
        )
        if additional_col_id2name is None:
            self.additional_col_id2name = {}
        else:
            assert all(
                isinstance(col_id, int) for col_id in additional_col_id2name.keys()
            )
            assert all(
                isinstance(col_name, str)
                for col_name in additional_col_id2name.values()
            )
            self.additional_col_id2name = additional_col_id2name

        assert sentence_sep_starts is None or all(
            isinstance(start, str) for start in sentence_sep_starts
        )
        self.sentence_sep_starts = sentence_sep_starts
        assert document_sep_starts is None or all(
            isinstance(start, str) for start in document_sep_starts
        )
        self.document_sep_starts = document_sep_starts
        super().__init__(
            is_tokenized=True, encoding=encoding, verbose=verbose, **token_kwargs
        )

    def read(self, file_path):
        doc_idx = 0
        data = []
        with open(file_path, "r", encoding=self.encoding) as f:
            text, tags = [], []
            additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}

            _is_skipping_last = True
            for line in f:
                line = line.strip()

                if self._is_document_seperator(line) or self._is_sentence_seperator(
                    line
                ):
                    if len(text) > 0:
                        additional_tags = {
                            self.additional_col_id2name[col_id]: atags
                            for col_id, atags in additional.items()
                        }
                        tokens = self._build_tokens(
                            text, additional_tags=additional_tags
                        )
                        chunks = self.tags_translator.tags2chunks(tags)
                        data.append(
                            {
                                "tokens": tokens,
                                "chunks": chunks,
                                "doc_idx": str(doc_idx),
                            }
                        )

                        text, tags = [], []
                        additional = {
                            col_id: [] for col_id in self.additional_col_id2name.keys()
                        }

                    if self._is_document_seperator(line):
                        # Empty documents will result in skipped indexes
                        doc_idx += 1

                    _is_skipping_last = True

                else:
                    line_seperated = line.split(self.sep)
                    text.append(line_seperated[self.text_col_id])
                    tags.append(line_seperated[self.tag_col_id])
                    for col_id in self.additional_col_id2name.keys():
                        additional[col_id].append(line_seperated[col_id])

                    # Fix for cases like ['I-ORG', '', 'I-ORG'], where the second is skipped.
                    if (
                        self.tags_translator.scheme == "BIO1"
                        and len(tags) >= 2
                        and tags[-1].startswith("I")
                        and tags[-1][1:] == tags[-2][1:]
                        and _is_skipping_last
                    ):
                        tags[-1] = tags[-1].replace("I", "B", 1)

                    _is_skipping_last = False

            if len(text) > 0:
                additional_tags = {
                    self.additional_col_id2name[col_id]: atags
                    for col_id, atags in additional.items()
                }
                tokens = self._build_tokens(text, additional_tags=additional_tags)
                chunks = self.tags_translator.tags2chunks(tags)
                data.append(
                    {"tokens": tokens, "chunks": chunks, "doc_idx": str(doc_idx)}
                )

        return data

    def _is_sentence_seperator(self, line: str):
        if line.strip() == "":
            return True
        if self.sentence_sep_starts is None:
            return False
        for start in self.sentence_sep_starts:
            if line.startswith(start):
                return True
        return False

    def _is_document_seperator(self, line: str):
        if self.document_sep_starts is None:
            return False
        for start in self.document_sep_starts:
            if line.startswith(start):
                return True
        return False

    def flatten_to_characters(self, data: list):
        additional_keys = [
            key
            for key in data[0]["tokens"][0].__dict__.keys()
            if key not in ("text", "raw_text")
        ]

        new_data = []
        for entry in data:
            tokenized_raw_text = entry["tokens"].raw_text
            char_seq_lens = [len(tok) for tok in tokenized_raw_text]
            cum_char_seq_lens = [0] + numpy.cumsum(char_seq_lens).tolist()

            flattened_tokenized_raw_text = [
                char for tok in tokenized_raw_text for char in tok
            ]
            # Repeat additional-tags for every character in a token
            flattened_additional_tags = {
                key: [
                    atag
                    for atag, tok in zip(
                        getattr(entry["tokens"], key), tokenized_raw_text
                    )
                    for char in tok
                ]
                for key in additional_keys
            }
            flattened_tokens = self._build_tokens(
                flattened_tokenized_raw_text, additional_tags=flattened_additional_tags
            )
            flattened_chunks = [
                (label, cum_char_seq_lens[start], cum_char_seq_lens[end])
                for label, start, end in entry["chunks"]
            ]
            new_data.append({"tokens": flattened_tokens, "chunks": flattened_chunks})

        return new_data
