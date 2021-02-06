# -*- coding: utf-8 -*-
from typing import List
import os
import logging
import torch

logger = logging.getLogger(__name__)


def _load_from_file(path: str, encoding=None):
    logger.info(f"Loading vectors from {path}")
    words = []
    vectors = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            w, *vector = line.split()
            words.append(w)
            vectors.append([float(v) for v in vector])
        
    vectors = torch.tensor(vectors)
    return words, vectors


class Vectors(object):
    def __init__(self, itos: List[str], vectors: torch.FloatTensor, unk_init=None):
        if len(itos) != vectors.size(0):
            raise ValueError(f"Vocaburaly size {len(itos)} does not match vector size {vectors.size(0)}")
            
        self.itos = itos
        self.vectors = vectors
        self.unk_init = torch.zeros if unk_init is None else unk_init
        
    def __getitem__(self, token: str):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(self.emb_dim)
        
        
    def lookup(self, token: str):
        tried_set = set()
        for possible_token in [token, token.lower(), token.title(), token.upper()]:
            if possible_token in tried_set:
                continue
            if possible_token in self.stoi:
                return self.vectors[self.stoi[possible_token]]
            else:
                tried_set.add(possible_token)
        return None
    
    
    @property
    def itos(self):
        return self._itos
    
    @itos.setter
    def itos(self, itos: List[str]):
        self._itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        
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
            itos, vectors = _load_from_file(path, encoding)
            cls.save_to_cache(path, itos, vectors)
        return cls(itos, vectors)
    
    
    
class GloVe(Vectors):
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
            with open(f"{path}/hash/words.lst", 'r') as f:
                itos = [w.strip() for w in f.readlines()]
            with open(f"{path}/embeddings/embeddings.txt", 'r') as f:
                vectors = [[float(v) for v in vector.strip().split()] for vector in f.readlines()]
                
            vectors = torch.tensor(vectors)
            self.save_to_cache(path, itos, vectors)
            
        super().__init__(itos, vectors, **kwargs)
            