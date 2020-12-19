# -*- coding: utf-8 -*-
import torch
from torchtext.experimental.vectors import Vectors


def Senna(path: str, unk_tensor=None):
    with open(f"{path}/hash/words.lst", 'r') as f:
        words = [w.strip() for w in f.readlines()]
    
    with open(f"{path}/embeddings/embeddings.txt", 'r') as f:
        vectors = [[float(v) for v in vector.strip().split()] for vector in f.readlines()]
        
    vectors = torch.tensor(vectors)
    return Vectors(words, vectors, unk_tensor=unk_tensor)
    
