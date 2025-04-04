# -*- coding: utf-8 -*-
import logging
import os
from typing import List, Union

import torch
import tqdm

logger = logging.getLogger(__name__)


def _parse_line(line: bytes):
    w, *vector = line.rstrip().split(b" ")
    return w, [float(v) for v in vector]


def _infer_shape(path: str, skiprows: List[int]):
    vec_dim = None
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i in skiprows:
                continue
            if vec_dim is None:
                w, vector = _parse_line(line)
                vec_dim = len(vector)

    return i + 1, vec_dim


def _load_from_file(
    path: str, encoding=None, skiprows: Union[int, List[int]] = None, verbose=False
):
    logger.info(f"Loading vectors from {path}")
    if skiprows is None:
        skiprows = []
    elif isinstance(skiprows, int):
        skiprows = [skiprows]
    assert all(isinstance(row, int) for row in skiprows)

    words, vectors = [], []
    num_lines, vec_dim = _infer_shape(path, skiprows)
    with open(path, "rb") as f:
        num_bad_lines = 0
        for i, line in tqdm.tqdm(
            enumerate(f),
            total=num_lines,
            disable=not verbose,
            ncols=100,
            desc="Loading vectors",
        ):
            if i in skiprows:
                continue
            try:
                w, vector = _parse_line(line)
                words.append(w.decode(encoding))
                assert len(vector) == vec_dim
                vectors.append(vector)
            except KeyboardInterrupt as e:
                raise e
            except:
                num_bad_lines += 1
                logger.warning(f"Bad line detected: {line.rstrip()}")

    if num_bad_lines > 0:
        logger.warning(f"Totally {num_bad_lines} bad lines exist and were skipped")

    vectors = torch.tensor(vectors)
    return words, vectors


class Vectors(object):
    def __init__(self, itos: List[str], vectors: torch.FloatTensor, unk_init=None):
        if len(itos) != vectors.size(0):
            raise ValueError(
                f"Vocaburaly size {len(itos)} does not match vector size {vectors.size(0)}"
            )

        self.itos = itos
        self.vectors = vectors
        self.unk_init = torch.zeros if unk_init is None else unk_init

    @property
    def itos(self):
        return self._itos

    @itos.setter
    def itos(self, itos: List[str]):
        self._itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __getitem__(self, token: str):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(self.emb_dim)

    def lookup(self, token: str):
        tried_set = set()
        # Backup tokens
        for possible_token in [token, token.lower(), token.title(), token.upper()]:
            if possible_token in tried_set:
                continue
            if possible_token in self.stoi:
                return self.vectors[self.stoi[possible_token]]
            else:
                tried_set.add(possible_token)
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.voc_dim}, {self.emb_dim})"

    def __len__(self):
        return self.vectors.size(0)

    @property
    def voc_dim(self):
        return self.vectors.size(0)

    @property
    def emb_dim(self):
        return self.vectors.size(1)

    @staticmethod
    def save_to_cache(path: str, itos: List[str], vectors: torch.FloatTensor):
        logger.info(f"Saving vectors to {path}.pt")
        torch.save((itos, vectors), f"{path}.pt")

    @staticmethod
    def load_from_cache(path: str):
        logger.info(f"Loading vectors from {path}.pt")
        itos, vectors = torch.load(f"{path}.pt")
        return itos, vectors

    @classmethod
    def load(cls, path: str, encoding=None, **kwargs):
        if os.path.exists(f"{path}.pt"):
            itos, vectors = cls.load_from_cache(path)
        else:
            itos, vectors = _load_from_file(path, encoding, **kwargs)
            cls.save_to_cache(path, itos, vectors)
        return cls(itos, vectors)


class GloVe(Vectors):
    """
    https://nlp.stanford.edu/projects/glove/
    """

    def __init__(self, path: str, encoding=None, **kwargs):
        if os.path.exists(f"{path}.pt"):
            itos, vectors = self.load_from_cache(path)
        else:
            itos, vectors = _load_from_file(path, encoding)
            self.save_to_cache(path, itos, vectors)

        super().__init__(itos, vectors, **kwargs)


class Senna(Vectors):
    def __init__(self, path: str, **kwargs):
        if os.path.exists(f"{path}.pt"):
            itos, vectors = self.load_from_cache(path)
        else:
            with open(f"{path}/hash/words.lst", "r") as f:
                itos = [w.strip() for w in f.readlines()]
            with open(f"{path}/embeddings/embeddings.txt", "r") as f:
                vectors = [
                    [float(v) for v in vector.strip().split()]
                    for vector in f.readlines()
                ]

            vectors = torch.tensor(vectors)
            self.save_to_cache(path, itos, vectors)

        super().__init__(itos, vectors, **kwargs)
