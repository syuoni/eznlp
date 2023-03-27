# -*- coding: utf-8 -*-
from typing import List


INV_REL_PREFIX = 'INV-'


def detect_missing_symmetric(relations: List[tuple], sym_rel_labels: List[str]): 
    missing_relations = []
    existing_chunk_pairs = set((head, tail) for _, head, tail in relations)
    for label, head, tail in relations: 
        if label in sym_rel_labels and (tail, head) not in existing_chunk_pairs: 
            missing_relations.append((label, tail, head))
    return missing_relations


def count_missing_symmetric(relations: List[tuple], sym_rel_labels: List[str]): 
    return len(detect_missing_symmetric(relations, sym_rel_labels))


def detect_inverse(relations: List[tuple]):
    inverse_relations = []
    existing_chunk_pairs = set((head, tail) for _, head, tail in relations)
    for label, head, tail in relations:
        if (tail, head) not in existing_chunk_pairs:
            inverse_relations.append((f"{INV_REL_PREFIX}{label}", tail, head))
    return inverse_relations


def count_inverse(relations: List[tuple]):
    return len(detect_inverse(relations))
