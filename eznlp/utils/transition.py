# -*- coding: utf-8 -*-
import os
import re
from collections import Counter
from typing import List

import pandas

from ..token import zh_char_re


class ChunksTagsTranslator(object):
    """The translator between chunks and tags.

    Args
    ----
    chunks: list
        list of (chunk_type, chunk_start, chunk_end).
    tags : list
        list of tags. e.g., ['O', 'B-Ent', 'I-Ent', ...].
    tokens : `TokenSequence`
        Token sequence object.

    References
    ----------
    https://github.com/chakki-works/seqeval
    """

    def __init__(self, scheme="BIOES", sep: str = "-", breaking_for_types: bool = True):
        assert scheme in ("BIO1", "BIO2", "BIOES", "BMES", "BILOU", "OntoNotes", "wwm")
        self.scheme = scheme

        dirname = os.path.dirname(__file__)
        sheet_name = "BIOES" if scheme in ("BMES", "BILOU") else scheme
        trans = pandas.read_excel(
            f"{dirname}/transition.xlsx",
            sheet_name=sheet_name,
            usecols=["from_tag", "to_tag", "legal", "end_of_chunk", "start_of_chunk"],
        )

        if scheme in ("BMES", "BILOU"):
            # Mapping from BIOES to BMES/BILOU
            if scheme == "BMES":
                mapper = {"B": "B", "I": "M", "O": "O", "E": "E", "S": "S"}
            elif scheme == "BILOU":
                mapper = {"B": "B", "I": "I", "O": "O", "E": "L", "S": "U"}
            trans["from_tag"] = trans["from_tag"].map(mapper)
            trans["to_tag"] = trans["to_tag"].map(mapper)

        trans = trans.set_index(["from_tag", "to_tag"])
        self.trans = {tr: trans.loc[tr].to_dict() for tr in trans.index.tolist()}
        self.sep = sep
        self.breaking_for_types = breaking_for_types

    def __repr__(self):
        return f"{self.__class__.__name__}(scheme={self.scheme})"

    def check_transitions_legal(self, tags: List[str]):
        """Check if the transitions between tags are legal."""
        # TODO: also check types
        padded_tags = (
            ["O"] + [tag.split(self.sep, maxsplit=1)[0] for tag in tags] + ["O"]
        )
        return all(
            [
                self.trans[(prev_tag, this_tag)]["legal"]
                for prev_tag, this_tag in zip(padded_tags[:-1], padded_tags[1:])
            ]
        )

    def chunks2group_by(self, chunks: List[tuple], seq_len: int):
        group_by = [-1 for _ in range(seq_len)]

        for i, (chunk_type, chunk_start, chunk_end) in enumerate(chunks):
            for j in range(chunk_start, chunk_end):
                group_by[j] = i

        return group_by

    def chunks2tags(self, chunks: List[tuple], seq_len: int):
        tags = ["O" for _ in range(seq_len)]

        # Longer chunks are of higher priority
        chunks = sorted(chunks, key=lambda ck: ck[2] - ck[1], reverse=True)

        for chunk_type, chunk_start, chunk_end in chunks:
            # Make sure the target slice is contained in the sequence
            # E.g., the sequence is truncated from a longer one
            # This causes unretrievable chunks
            if chunk_start < 0 or chunk_end > seq_len:
                continue

            # Make sure the target slice has not been labeled
            # E.g., nested entities
            # This causes unretrievable chunks
            if not all(tags[k] == "O" for k in range(chunk_start, chunk_end)):
                continue

            if self.scheme == "BIO1":
                if chunk_start == 0 or tags[chunk_start - 1] == "O":
                    tags[chunk_start] = f"I{self.sep}{chunk_type}"
                else:
                    tags[chunk_start] = f"B{self.sep}{chunk_type}"
                for k in range(chunk_start + 1, chunk_end):
                    tags[k] = f"I{self.sep}{chunk_type}"
                if chunk_end < len(tags) and tags[chunk_end].startswith(f"I{self.sep}"):
                    tags[chunk_end] = tags[chunk_end].replace(
                        f"I{self.sep}", f"B{self.sep}", 1
                    )

            elif self.scheme == "BIO2":
                tags[chunk_start] = f"B{self.sep}{chunk_type}"
                for k in range(chunk_start + 1, chunk_end):
                    tags[k] = f"I{self.sep}{chunk_type}"

            elif self.scheme == "BIOES":
                if chunk_end - chunk_start == 1:
                    tags[chunk_start] = f"S{self.sep}{chunk_type}"
                else:
                    tags[chunk_start] = f"B{self.sep}{chunk_type}"
                    tags[chunk_end - 1] = f"E{self.sep}{chunk_type}"
                    for k in range(chunk_start + 1, chunk_end - 1):
                        tags[k] = f"I{self.sep}{chunk_type}"

            elif self.scheme == "OntoNotes":
                if chunk_end - chunk_start == 1:
                    tags[chunk_start] = f"({chunk_type})"
                else:
                    tags[chunk_start] = f"({chunk_type}*"
                    tags[chunk_end - 1] = "*)"
                    for k in range(chunk_start + 1, chunk_end - 1):
                        tags[k] = "*"

        if self.scheme == "OntoNotes":
            tags = ["*" if tag == "O" else tag for tag in tags]

        return tags

    def _vote_in_types(self, chunk_types):
        if self.breaking_for_types:
            # All elements must be the same.
            return chunk_types[0]
        else:
            # Assign the first element with 0.5 higher weight.
            type_counter = Counter(chunk_types)
            type_counter[chunk_types[0]] += 0.5
            return type_counter.most_common(1)[0][0]

    def tags2chunks(self, tags: List[str]):
        if self.scheme == "OntoNotes":
            return self.ontonotes_tags2chunks(tags)

        chunks = []
        prev_tag, prev_type = "O", "O"
        chunk_start, chunk_types = -1, []

        for k, tag in enumerate(tags):
            if tag in ("O", "<pad>"):
                this_tag, this_type = "O", "O"
            else:
                if self.sep in tag:
                    this_tag, this_type = tag.split(self.sep, maxsplit=1)
                else:
                    # Typically cascade-tags without types
                    this_tag, this_type = tag, "<pseudo-type>"

            this_trans = self.trans[(prev_tag, this_tag)]
            is_in_chunk = (
                (prev_tag != "O")
                and (this_tag != "O")
                and (not this_trans["end_of_chunk"])
                and (not this_trans["start_of_chunk"])
            )

            # Breaking because of different types, is holding only in case of `is_in_chunk` being True.
            # In such case, the `prev_tag` must be `B` or `I` and
            #               the `this_tag` must be `I` or `E`.
            # The breaking operation is equivalent to treating `this_tag` as `B`.
            if is_in_chunk and self.breaking_for_types and (this_type != prev_type):
                this_trans = self.trans[(prev_tag, "B")]
                is_in_chunk = False

            if this_trans["end_of_chunk"]:
                chunks.append((self._vote_in_types(chunk_types), chunk_start, k))
                chunk_types = []

            if this_trans["start_of_chunk"]:
                chunk_start = k
                chunk_types = [this_type]

            if is_in_chunk:
                chunk_types.append(this_type)

            prev_tag, prev_type = this_tag, this_type

        if prev_tag != "O":
            chunks.append((self._vote_in_types(chunk_types), chunk_start, len(tags)))

        return chunks

    def ontonotes_tags2chunks(self, tags: List[str]):
        chunks = []
        prev_tag = "*)"
        chunk_start, chunk_type = -1, None

        for k, tag in enumerate(tags):
            this_tag = "".join(re.findall("[\(\*\)]", tag))
            this_type = re.sub("[\(\*\)]", "", tag)

            this_trans = self.trans[(prev_tag, this_tag)]

            if this_trans["end_of_chunk"] and (chunk_type is not None):
                chunks.append((chunk_type, chunk_start, k))
                chunk_type = None

            if this_trans["start_of_chunk"]:
                chunk_start = k
                chunk_type = this_type

            prev_tag = this_tag

        if self.trans[(prev_tag, "(*")]["end_of_chunk"]:
            chunks.append((chunk_type, chunk_start, len(tags)))

        return chunks


def _token2wwm_tag(tok: str, subword_prefix: str = "##"):
    if tok.startswith("[") and tok.endswith("]"):
        return "SP-SP"

    # Theoretically, Chinese characters never follow `##`
    # Chinese punctuations belong to `ETC`, and may follow `##`

    if tok.startswith(subword_prefix):
        if tok[2:].isascii():
            return "##EN-EN"  # The returning prefix `##` corresponds to that in `transition.xlsx`
        elif zh_char_re.fullmatch(tok[2:]):  # zh_punct_re.fullmatch(tok[2:])
            return "##ZH-ZH"
        else:
            return "##ETC-ETC"
    else:
        if tok.isascii():
            return "EN-EN"
        elif zh_char_re.fullmatch(tok):
            return "ZH-ZH"
        else:
            return "ETC-ETC"
