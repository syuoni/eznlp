# -*- coding: utf-8 -*-
import json
import logging
import os
from typing import List

from ..utils import TextChunksTranslator
from .base import IO

logger = logging.getLogger(__name__)


def _filter_duplicated(tuples: List[tuple]):
    filtered_tuples = []
    for tp in tuples:
        if tp not in filtered_tuples:
            filtered_tuples.append(tp)
    return filtered_tuples


# TODO: rename as InfoExIO?
class JsonIO(IO):
    """An IO Interface of Json files."""

    def __init__(
        self,
        is_tokenized: bool = True,
        tokenize_callback=None,
        text_key="tokens",
        chunk_key="entities",
        chunk_type_key="type",
        chunk_start_key="start",
        chunk_end_key="end",
        chunk_text_key=None,
        attribute_key=None,
        attribute_type_key=None,
        attribute_chunk_key=None,
        relation_key=None,
        relation_type_key=None,
        relation_head_key=None,
        relation_tail_key=None,
        drop_duplicated=True,
        is_whole_piece: bool = True,
        retain_keys: list = None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        self.text_key = text_key
        self.chunk_key = chunk_key
        self.chunk_type_key = chunk_type_key
        self.chunk_start_key = chunk_start_key
        self.chunk_end_key = chunk_end_key
        self.chunk_text_key = chunk_text_key

        if all(
            key is not None
            for key in [attribute_key, attribute_type_key, attribute_chunk_key]
        ):
            self.attribute_key = attribute_key
            self.attribute_type_key = attribute_type_key
            self.attribute_chunk_key = attribute_chunk_key
        else:
            self.attribute_key = None

        if all(
            key is not None
            for key in [
                relation_key,
                relation_type_key,
                relation_head_key,
                relation_tail_key,
            ]
        ):
            self.relation_key = relation_key
            self.relation_type_key = relation_type_key
            self.relation_head_key = relation_head_key
            self.relation_tail_key = relation_tail_key
        else:
            self.relation_key = None

        self.drop_duplicated = drop_duplicated
        self.is_whole_piece = is_whole_piece
        self.retain_keys = [] if retain_keys is None else retain_keys
        assert all(
            key not in self.retain_keys
            for key in [
                self.text_key,
                self.chunk_key,
                self.relation_key,
                self.attribute_key,
            ]
        )

        super().__init__(
            is_tokenized=is_tokenized,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )
        if not self.is_tokenized:
            self.text_translator = TextChunksTranslator()

    def read(self, file_path, return_errors: bool = False):
        with open(file_path, "r", encoding=self.encoding) as f:
            if self.is_whole_piece:
                raw_data = json.load(f)
            else:
                raw_data = [json.loads(line) for line in f if len(line.strip()) > 0]

        data = []
        errors, mismatches = [], []
        for raw_entry in raw_data:
            tokens = self._build_tokens(raw_entry[self.text_key])
            chunks = [
                (
                    chunk[self.chunk_type_key],
                    chunk[self.chunk_start_key],
                    chunk[self.chunk_end_key],
                )
                for chunk in raw_entry[self.chunk_key]
            ]

            if not self.is_tokenized:
                if self.chunk_text_key is not None:
                    chunks = [
                        (*ck, chunk[self.chunk_text_key])
                        for ck, chunk in zip(chunks, raw_entry[self.chunk_key])
                    ]

                (
                    chunks,
                    curr_errors,
                    curr_mismatches,
                ) = self.text_translator.text_chunks2chunks(
                    chunks, tokens, raw_entry[self.text_key], place_none_for_errors=True
                )
                errors.extend(curr_errors)
                mismatches.extend(curr_mismatches)

            entry = {
                "tokens": tokens,
                "chunks": [ck for ck in chunks if ck is not None],
            }

            if self.attribute_key is not None:
                attributes = [
                    (
                        attr[self.attribute_type_key],
                        chunks[attr[self.attribute_chunk_key]],
                    )
                    for attr in raw_entry[self.attribute_key]
                ]
                entry.update(
                    {
                        "attributes": [
                            (attr_type, ck)
                            for attr_type, ck in attributes
                            if ck is not None
                        ]
                    }
                )

            if self.relation_key is not None:
                relations = [
                    (
                        rel[self.relation_type_key],
                        chunks[rel[self.relation_head_key]],
                        chunks[rel[self.relation_tail_key]],
                    )
                    for rel in raw_entry[self.relation_key]
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

            if self.drop_duplicated:
                entry["chunks"] = _filter_duplicated(entry["chunks"])
                if self.attribute_key is not None:
                    entry["attributes"] = _filter_duplicated(entry["attributes"])
                if self.relation_key is not None:
                    entry["relations"] = _filter_duplicated(entry["relations"])

            entry.update({k: v for k, v in raw_entry.items() if k in self.retain_keys})
            data.append(entry)

        if len(errors) > 0 or len(mismatches) > 0:
            logger.warning(
                f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing {file_path}"
            )

        if return_errors:
            return data, errors, mismatches
        else:
            return data

    def write(self, data: List[dict], file_path):
        raw_data = []
        for entry in data:
            raw_entry = {self.text_key: entry["tokens"].raw_text}

            chunk2idx = {ck: k for k, ck in enumerate(entry["chunks"])}
            raw_entry[self.chunk_key] = [
                {
                    self.chunk_type_key: chunk_type,
                    self.chunk_start_key: chunk_start,
                    self.chunk_end_key: chunk_end,
                }
                for chunk_type, chunk_start, chunk_end in entry["chunks"]
            ]
            if self.attribute_key is not None:
                raw_entry[self.attribute_key] = [
                    {
                        self.attribute_type_key: attr_type,
                        self.attribute_chunk_key: chunk2idx[ck],
                    }
                    for attr_type, ck in entry["attributes"]
                ]
            if self.relation_key is not None:
                raw_entry[self.relation_key] = [
                    {
                        self.relation_type_key: rel_type,
                        self.relation_head_key: chunk2idx[head],
                        self.relation_tail_key: chunk2idx[tail],
                    }
                    for rel_type, head, tail in entry["relations"]
                ]

            raw_entry.update({k: v for k, v in entry.items() if k in self.retain_keys})
            raw_data.append(raw_entry)

        with open(file_path, "w", encoding=self.encoding) as f:
            if self.is_whole_piece:
                json.dump(raw_data, f, ensure_ascii=False)
            else:
                for raw_entry in raw_data:
                    f.write(json.dumps(raw_entry, ensure_ascii=False))
                    f.write("\n")


class SQuADIO(IO):
    """An IO Interface of SQuAD files."""

    def __init__(
        self,
        tokenize_callback=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        super().__init__(
            is_tokenized=False,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )
        self.text_translator = TextChunksTranslator()

    def read(self, file_path, return_errors: bool = False):
        with open(file_path, "r", encoding=self.encoding) as f:
            raw_data = json.load(f)
        raw_data = raw_data["data"]

        data = []
        errors, mismatches = [], []
        for doc in raw_data:
            for parag in doc["paragraphs"]:
                context = self._build_tokens(parag["context"])

                for qas in parag["qas"]:
                    assert qas["is_impossible"] or len(qas["answers"]) > 0

                    question = self._build_tokens(qas["question"])
                    answers = [
                        (
                            "<ans>",
                            ans["answer_start"],
                            ans["answer_start"] + len(ans["text"]),
                            ans["text"],
                        )
                        for ans in qas["answers"]
                    ]
                    (
                        answers,
                        curr_errors,
                        curr_mismatches,
                    ) = self.text_translator.text_chunks2chunks(
                        answers, context, parag["context"]
                    )
                    answers = _filter_duplicated(answers)
                    errors.extend(curr_errors)
                    mismatches.extend(curr_mismatches)
                    data.append(
                        {
                            "id": qas["id"],
                            "title": doc["title"],
                            "context": context,
                            "question": question,
                            "answers": answers,
                        }
                    )

        if len(errors) > 0 or len(mismatches) > 0:
            logger.warning(
                f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing {file_path}"
            )

        if return_errors:
            return data, errors, mismatches
        else:
            return data


class KarpathyIO(IO):
    """An IO Interface of json files by Karpathy et al. (2015).

    References
    ----------
    [1] Karpathy and Li. 2015. Deep visual-semantic alignments for generating image descriptions. CVPR, 2015.
    """

    def __init__(
        self,
        img_folder: str,
        check_img_path: bool = False,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        self.img_folder = img_folder
        self.check_img_path = check_img_path
        super().__init__(
            is_tokenized=True,
            tokenize_callback=None,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )

    def read(self, file_path):
        with open(file_path, "r", encoding=self.encoding) as f:
            raw_data = json.load(f)
        raw_data = raw_data["images"]

        train_data, dev_data, test_data = [], [], []
        for raw_entry in raw_data:
            entry = {
                "img_path": f"{self.img_folder}/{raw_entry['filename']}",
                "full_trg_tokens": [
                    self._build_tokens(sent["tokens"])
                    for sent in raw_entry["sentences"]
                ],
            }

            if self.check_img_path:
                assert os.path.exists(entry["img_path"])

            if raw_entry["split"].lower().startswith(("dev", "val")):
                dev_data.append(entry)
            elif raw_entry["split"].lower().startswith("test"):
                test_data.append(entry)
            else:
                train_data.append(entry)

        return train_data, dev_data, test_data


class TextClsIO(IO):
    """An IO interface of (single or paired) text classification dataset in json files."""

    def __init__(
        self,
        is_tokenized: bool = False,
        tokenize_callback=None,
        text_key: str = "text",
        paired_text_key: str = None,
        label_key="label",
        is_whole_piece: bool = True,
        retain_keys: list = None,
        mapping=None,
        encoding=None,
        verbose: bool = True,
        **token_kwargs,
    ):
        self.text_key = text_key
        self.paired_text_key = paired_text_key
        self.label_key = label_key
        self.is_whole_piece = is_whole_piece
        self.retain_keys = [] if retain_keys is None else retain_keys
        assert all(
            key not in self.retain_keys
            for key in [self.text_key, self.paired_text_key, self.label_key]
        )
        self.mapping = {} if mapping is None else mapping
        super().__init__(
            is_tokenized=is_tokenized,
            tokenize_callback=tokenize_callback,
            encoding=encoding,
            verbose=verbose,
            **token_kwargs,
        )

    def read(self, file_path):
        with open(file_path, "r", encoding=self.encoding) as f:
            if self.is_whole_piece:
                raw_data = json.load(f)
            else:
                raw_data = [json.loads(line) for line in f if len(line.strip()) > 0]

        data = []
        for raw_entry in raw_data:
            raw_text = raw_entry[self.text_key]
            for pattern, repl in self.mapping.items():
                raw_text = raw_text.replace(pattern, repl)
            entry = {"tokens": self._build_tokens(raw_text)}

            if self.paired_text_key is not None:
                paired_raw_text = raw_entry[self.paired_text_key]
                for pattern, repl in self.mapping.items():
                    paired_raw_text = paired_raw_text.replace(pattern, repl)
                entry.update({"paired_tokens": self._build_tokens(paired_raw_text)})

            if self.label_key in raw_entry:
                entry.update({"label": raw_entry[self.label_key]})

            entry.update({k: v for k, v in raw_entry.items() if k in self.retain_keys})
            data.append(entry)

        return data
