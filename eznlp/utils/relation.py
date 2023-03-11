# -*- coding: utf-8 -*-
from typing import List


def detect_missing_symmetric(relations: List[tuple], sym_rel_labels: List[str]): 
    missing_relations = []
    for label, head, tail in relations: 
        if label in sym_rel_labels: 
            if (label, tail, head) not in relations: 
                missing_relations.append((label, tail, head))
    return missing_relations


def count_missing_symmetric(relations: List[tuple], sym_rel_labels: List[str]): 
    return len(detect_missing_symmetric(relations, sym_rel_labels))
