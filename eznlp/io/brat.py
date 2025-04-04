# -*- coding: utf-8 -*-
import glob
import logging
import os
import re
from typing import List

import numpy

from ..utils import TextChunksTranslator
from ..utils.segmentation import (
    segment_text_uniformly,
    segment_text_with_hierarchical_seps,
)
from .base import IO

logger = logging.getLogger(__name__)


class BratIO(IO):
    """An IO interface of brat-format files.

    Notes
    -----
    For Chinese text with inserted spaces, `ins_space_tokenize_callback` is the word-level tokenizer,
    and `tokenize_callback` is the character-level tokenizer.

    If `max_len` is specified, it will segment the text by `hie_seps` such that the segments are as long as possible to be close to `max_len`;
    If `max_len` is not specified (`None`), it will segment the text by `hie_seps` such that the segments are as short as possible.
    """

    def __init__(
        self,
        tokenize_callback="char",
        has_ins_space: bool = False,
        ins_space_tokenize_callback=None,
        parse_attrs: bool = False,
        parse_relations: bool = False,
        max_len=500,
        line_sep="\r\n",
        sentence_seps=None,
        phrase_seps=None,
        allow_broken_chunk_text: bool = False,
        consistency_mapping: dict = None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        self.has_ins_space = has_ins_space
        self.ins_space_tokenize_callback = ins_space_tokenize_callback
        assert not (self.has_ins_space and self.ins_space_tokenize_callback is None)
        self.inserted_mark = "⬜"

        self.parse_attrs = parse_attrs
        self.parse_relations = parse_relations

        self.max_len = max_len
        self.line_sep = line_sep
        self.sentence_seps = ["。"] if sentence_seps is None else sentence_seps
        self.phrase_seps = ["；", "，", ";", ","] if phrase_seps is None else phrase_seps
        self.hie_seps = [[self.line_sep], self.sentence_seps, self.phrase_seps]

        self.allow_broken_chunk_text = allow_broken_chunk_text

        super().__init__(
            is_tokenized=False,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )
        self.text_translator = TextChunksTranslator(
            consistency_mapping=consistency_mapping
        )

    def _parse_text_chunk_ann(self, ann: str):
        chunk_id, chunk_type_pos, chunk_text = ann.rstrip(self.line_sep).split("\t")
        if ";" in chunk_type_pos:
            # Potential non-continuous chunks
            (
                chunk_type,
                chunk_start_in_text,
                *_,
                chunk_end_in_text,
            ) = chunk_type_pos.split(" ")
        else:
            chunk_type, chunk_start_in_text, chunk_end_in_text = chunk_type_pos.split(
                " "
            )
        chunk_start_in_text = int(chunk_start_in_text)
        chunk_end_in_text = int(chunk_end_in_text)
        return chunk_id, (
            chunk_type,
            chunk_start_in_text,
            chunk_end_in_text,
            chunk_text,
        )

    def _build_text_chunk_ann(self, chunk_id: str, text_chunk: tuple):
        chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text = text_chunk
        return f"{chunk_id}\t{chunk_type} {chunk_start_in_text} {chunk_end_in_text}\t{chunk_text}"

    def _parse_attr_ann(self, ann: str):
        attr_id, attr_name_and_chunk_id = ann.rstrip(self.line_sep).split("\t")
        attr_name, chunk_id = attr_name_and_chunk_id.split(" ")
        return attr_id, chunk_id, attr_name

    def _build_attr_ann(self, attr_id: str, chunk_id: str, attr_name: str):
        return f"{attr_id}\t{attr_name} {chunk_id}"

    def _parse_relation_ann(self, ann: str):
        rel_id, rel_type_and_chunk_ids, *_ = ann.rstrip(self.line_sep).split("\t")
        rel_type, head_id, tail_id = rel_type_and_chunk_ids.split(" ")
        _, head_id = head_id.split(":")
        _, tail_id = tail_id.split(":")
        return rel_id, head_id, tail_id, rel_type

    def _build_relation_ann(
        self, rel_id: str, head_id: str, tail_id: str, rel_type: str
    ):
        return f"{rel_id}\t{rel_type} Arg1:{head_id} Arg2:{tail_id}\t"

    def _check_text_chunks(self, text: str, text_chunks: dict):
        # Check chunk-text consistenty
        for chunk_id, text_chunk in text_chunks.items():
            chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text = text_chunk
            chunk_text_retr = text[chunk_start_in_text:chunk_end_in_text]

            if self.allow_broken_chunk_text:
                if not self.text_translator.is_consistency(chunk_text, chunk_text_retr):
                    logger.warning(
                        f"Inconsistent chunk text detected: {chunk_text} vs. {chunk_text_retr}"
                    )
            else:
                assert self.text_translator.is_consistency(chunk_text, chunk_text_retr)

    def _clean_text_chunks(self, text: str, text_chunks: dict):
        # Replace `"\t"` with `" "`
        text = re.sub("[ \t]", " ", text)
        for chunk_id, text_chunk in text_chunks.items():
            chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text = text_chunk
            chunk_text = re.sub("[ \t]", " ", chunk_text)
            text_chunks[chunk_id] = (
                chunk_type,
                chunk_start_in_text,
                chunk_end_in_text,
                chunk_text,
            )
        return text, text_chunks

    def _segment_text(self, text: str):
        for start, end in segment_text_with_hierarchical_seps(
            text, hie_seps=self.hie_seps, length=self.max_len
        ):
            if self.max_len is None or end - start <= self.max_len:
                yield (start, end)
            else:
                for sub_start, sub_end in segment_text_uniformly(
                    text[start:end], max_span_size=self.max_len
                ):
                    yield (start + sub_start, start + sub_end)

    def _fix_broken_chunk_text(self, anns: List[str]):
        new_anns = []
        for ann in anns:
            if ann.startswith(("T", "A", "R", "*", "#")):
                new_anns.append(ann)
            else:
                assert new_anns[-1].startswith("T")
                logger.info(
                    f"Fixing annotation: {new_anns[-1].strip()} <- {ann.strip()}"
                )

                chunk_id, (
                    chunk_type,
                    chunk_start_in_text,
                    chunk_end_in_text,
                    chunk_text,
                ) = self._parse_text_chunk_ann(new_anns[-1] + ann)
                chunk_end_in_text = chunk_start_in_text + len(chunk_text)
                new_anns[-1] = self._build_text_chunk_ann(
                    chunk_id,
                    (chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text),
                )

        return new_anns

    def read(self, file_path, return_errors: bool = False):
        with open(file_path, "r", encoding=self.encoding) as f:
            text = f.read()
        with open(file_path.replace(".txt", ".ann"), "r", encoding=self.encoding) as f:
            anns = [ann for ann in f.readlines() if not ann.strip() == ""]
            if self.allow_broken_chunk_text:
                anns = self._fix_broken_chunk_text(anns)
            assert all(ann.startswith(("T", "A", "R", "*", "#")) for ann in anns)

        # Restore "\n" -> "\r\n"
        if ("\n" in text) and (self.line_sep not in text):
            text = text.replace("\n", self.line_sep)

        # Parse chunks
        text_chunks = dict(
            [self._parse_text_chunk_ann(ann) for ann in anns if ann.startswith("T")]
        )

        # Clean and check chunks
        text, text_chunks = self._clean_text_chunks(text, text_chunks)
        self._check_text_chunks(text, text_chunks)
        if self.has_ins_space:
            text, text_chunks = self._remove_inserted_spaces(text, text_chunks)
            self._check_text_chunks(text, text_chunks)

        # Parse attributes
        if self.parse_attrs:
            text_attrs = [
                self._parse_attr_ann(ann) for ann in anns if ann.startswith("A")
            ]

        # Parse relations
        if self.parse_relations:
            text_relations = [
                self._parse_relation_ann(ann) for ann in anns if ann.startswith("R")
            ]

        data = []
        errors, mismatches = [], []
        for span_start_in_text, span_end_in_text in self._segment_text(text):
            curr_text = text[span_start_in_text:span_end_in_text]

            if len(curr_text.strip()) > 0:
                tokens = self._build_tokens(curr_text)
                curr_text_chunks = {
                    chunk_id: (
                        chunk_type,
                        chunk_start_in_text - span_start_in_text,
                        chunk_end_in_text - span_start_in_text,
                        chunk_text,
                    )
                    for chunk_id, (
                        chunk_type,
                        chunk_start_in_text,
                        chunk_end_in_text,
                        chunk_text,
                    ) in text_chunks.items()
                    if span_start_in_text <= chunk_start_in_text
                    and chunk_end_in_text <= span_end_in_text
                }

                curr_idx2chunk_id = list(curr_text_chunks.keys())
                curr_chunk_id2idx = {
                    chunk_id: idx for idx, chunk_id in enumerate(curr_idx2chunk_id)
                }

                (
                    curr_chunks,
                    curr_errors,
                    curr_mismatches,
                ) = self.text_translator.text_chunks2chunks(
                    [curr_text_chunks[chunk_id] for chunk_id in curr_idx2chunk_id],
                    tokens,
                    curr_text,
                    place_none_for_errors=True,
                )
                assert len(curr_chunks) == len(curr_text_chunks)
                errors.extend(curr_errors)
                mismatches.extend(curr_mismatches)
                entry = {
                    "tokens": tokens,
                    "chunks": [ck for ck in curr_chunks if ck is not None],
                }

                if self.parse_attrs:
                    curr_attrs = [
                        (attr_name, curr_chunks[curr_chunk_id2idx[chunk_id]])
                        for attr_id, chunk_id, attr_name in text_attrs
                        if chunk_id in curr_idx2chunk_id
                    ]
                    entry.update(
                        {
                            "attributes": [
                                (attr_name, ck)
                                for attr_name, ck in curr_attrs
                                if ck is not None
                            ]
                        }
                    )

                if self.parse_relations:
                    relations = [
                        (
                            rel_type,
                            curr_chunks[curr_chunk_id2idx[head_id]],
                            curr_chunks[curr_chunk_id2idx[tail_id]],
                        )
                        for rel_id, head_id, tail_id, rel_type in text_relations
                        if head_id in curr_idx2chunk_id and tail_id in curr_idx2chunk_id
                    ]
                    entry.update(
                        {
                            "relations": [
                                (rel_type, head, tail)
                                for rel_type, head, tail in relations
                                if head is not None and tail is not None
                            ]
                        }
                    )

                data.append(entry)

        if len(errors) > 0 or len(mismatches) > 0:
            logger.warning(
                f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing {file_path}"
            )

        if return_errors:
            return data, errors, mismatches
        else:
            return data

    def read_files(self, file_paths, return_errors: bool = False):
        data = []
        errors, mismatches = [], []
        for file_path in file_paths:
            curr_data, curr_errors, curr_mismatches = self.read(
                file_path, return_errors=True
            )
            data.extend(curr_data)
            errors.extend(curr_errors)
            mismatches.extend(curr_mismatches)

        if return_errors:
            return data, errors, mismatches
        else:
            return data

    def read_folder(self, folder_path, return_errors: bool = False):
        file_paths = [
            file_path
            for file_path in glob.iglob(f"{folder_path}/*.txt")
            if os.path.exists(file_path.replace(".txt", ".ann"))
        ]
        return self.read_files(file_paths, return_errors=return_errors)

    def write(self, data: List[dict], file_path):
        text_spans = []
        chunk_idx, attr_idx, rel_idx = 1, 1, 1
        text_chunks, attr_anns, rel_anns = {}, [], []

        span_start_in_text = 0
        for entry in data:
            tokens, curr_chunks = entry["tokens"], entry["chunks"]

            curr_text = tokens.to_raw_text()
            text_spans.append(curr_text)

            curr_text_chunks = self.text_translator.chunks2text_chunks(
                curr_chunks, tokens, curr_text, append_chunk_text=True
            )
            curr_idx2chunk_id = [f"T{chunk_idx+k}" for k in range(len(curr_chunks))]

            text_chunks.update(
                {
                    chunk_id: (
                        chunk_type,
                        chunk_start_in_text + span_start_in_text,
                        chunk_end_in_text + span_start_in_text,
                        chunk_text,
                    )
                    for chunk_id, (
                        chunk_type,
                        chunk_start_in_text,
                        chunk_end_in_text,
                        chunk_text,
                    ) in zip(curr_idx2chunk_id, curr_text_chunks)
                }
            )
            chunk_idx += len(entry["chunks"])
            span_start_in_text += len(curr_text) + len(self.line_sep)

            if self.parse_attrs:
                curr_attr_anns = [
                    self._build_attr_ann(
                        f"A{attr_idx+k}",
                        curr_idx2chunk_id[curr_chunks.index(chunk)],
                        attr_name,
                    )
                    for k, (attr_name, chunk) in enumerate(entry["attributes"])
                ]
                attr_anns.extend(curr_attr_anns)
                attr_idx += len(entry["attributes"])

            if self.parse_relations:
                curr_rel_anns = [
                    self._build_relation_ann(
                        f"R{rel_idx+k}",
                        curr_idx2chunk_id[curr_chunks.index(head)],
                        curr_idx2chunk_id[curr_chunks.index(tail)],
                        rel_type,
                    )
                    for k, (rel_type, head, tail) in enumerate(entry["relations"])
                ]
                rel_anns.extend(curr_rel_anns)
                rel_idx += len(entry["relations"])

        text = self.line_sep.join(text_spans)

        # Clean and check chunks
        text, text_chunks = self._clean_text_chunks(text, text_chunks)
        self._check_text_chunks(text, text_chunks)
        if self.has_ins_space:
            text, text_chunks = self._insert_spaces(text, text_chunks)
            self._check_text_chunks(text, text_chunks)

        text_chunk_anns = [
            self._build_text_chunk_ann(chunk_id, text_chunk)
            for chunk_id, text_chunk in text_chunks.items()
        ]

        # Map "\r\n" -> "\n"
        with open(file_path, "w", encoding=self.encoding) as f:
            f.write(text.replace(self.line_sep, "\n"))
            f.write("\n")
        with open(file_path.replace(".txt", ".ann"), "w", encoding=self.encoding) as f:
            f.write("\n".join(text_chunk_anns))
            f.write("\n")
            if self.parse_attrs:
                f.write("\n".join(attr_anns))
                f.write("\n")
            if self.parse_relations:
                f.write("\n".join(rel_anns))
                f.write("\n")

    def _tokenize_and_rejoin(self, text: str):
        tokenized = []
        for tok in self.ins_space_tokenize_callback(text):
            if (not re.fullmatch("\s+", tok)) and (len(tokenized) > 0):
                tokenized.append(self.inserted_mark)
            tokenized.append(tok)

        text = "".join(tokenized)
        text = re.sub(
            f"[“/]{self.inserted_mark}(?!\s)",
            lambda x: x.group().replace(self.inserted_mark, ""),
            text,
        )
        text = re.sub(
            f"(?<!\s){self.inserted_mark}[”：/]",
            lambda x: x.group().replace(self.inserted_mark, ""),
            text,
        )
        return text

    def _insert_spaces(self, text: str, text_chunks: dict):
        """Insert spaces for display Chinese text in brat UI.

        Notes
        -----
        Chinese text may be tokenized and re-joined by spaces before annotation.

        Any `N` consecutive spaces (in the original text) would be inserted one
        additional space, resulting in `N+1` consecutive spaces.
        """
        ori_num_chars = len(text)
        text = self.line_sep.join(
            [self._tokenize_and_rejoin(line) for line in text.split(self.line_sep)]
        )

        is_inserted = [int(c == self.inserted_mark) for c in text]
        num_inserted = numpy.cumsum(is_inserted).tolist()
        num_inserted = [n for n, i in zip(num_inserted, is_inserted) if i == 0]
        assert len(num_inserted) == ori_num_chars
        num_inserted.append(num_inserted[-1])

        text = text.replace(self.inserted_mark, " ")
        for chunk_id, text_chunk in text_chunks.items():
            chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text = text_chunk
            chunk_start_in_text += num_inserted[chunk_start_in_text]
            chunk_end_in_text += num_inserted[chunk_end_in_text] - int(
                num_inserted[chunk_end_in_text] > num_inserted[chunk_end_in_text - 1]
            )
            assert text[chunk_start_in_text:chunk_end_in_text].replace(
                " ", ""
            ) == chunk_text.replace(" ", "")
            chunk_text = text[chunk_start_in_text:chunk_end_in_text]

            text_chunks[chunk_id] = (
                chunk_type,
                chunk_start_in_text,
                chunk_end_in_text,
                chunk_text,
            )

        return text, text_chunks

    def _remove_inserted_spaces(self, text: str, text_chunks: dict):
        """Remove the spaces which are inserted for display Chinese text in brat UI.

        Notes
        -----
        Chinese text may be tokenized and re-joined by spaces for better display before annotation.

        Any single space would be regarded as a inserted space;
        Any `N` consecutive spaces would be regared as one space (in the original text) with `N-1` inserted spaces.
        """
        text = re.sub("(?<! ) (?! )", self.inserted_mark, text)
        text = re.sub(
            " {2,}", lambda x: " " + self.inserted_mark * (len(x.group()) - 1), text
        )

        is_inserted = [int(c == self.inserted_mark) for c in text]
        num_inserted = numpy.cumsum(is_inserted).tolist()
        # The positions of exact pre-inserted spaces should be mapped to positions NEXT to them
        num_inserted = [n - i for n, i in zip(num_inserted, is_inserted)]
        num_inserted.append(num_inserted[-1])

        text = text.replace(self.inserted_mark, "")
        for chunk_id, text_chunk in text_chunks.items():
            chunk_type, chunk_start_in_text, chunk_end_in_text, chunk_text = text_chunk
            chunk_start_in_text -= num_inserted[chunk_start_in_text]
            chunk_end_in_text -= num_inserted[chunk_end_in_text]
            assert text[chunk_start_in_text:chunk_end_in_text].replace(
                " ", ""
            ) == chunk_text.replace(" ", "")
            chunk_text = text[chunk_start_in_text:chunk_end_in_text]

            text_chunks[chunk_id] = (
                chunk_type,
                chunk_start_in_text,
                chunk_end_in_text,
                chunk_text,
            )

        return text, text_chunks
