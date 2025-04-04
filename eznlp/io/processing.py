# -*- coding: utf-8 -*-
from typing import Callable, List, Union

import tqdm


def _make_tuple_mapping(
    type_mapping: Union[Callable, dict] = None, aux_check: Callable = None
):
    def tuple_mapping(x):
        if aux_check is not None and not aux_check(x):
            return None

        if type_mapping is None:
            return x

        if callable(type_mapping):
            x_type = type_mapping(x[0])
        elif isinstance(type_mapping, dict):
            x_type = type_mapping.get(x[0], None)
        return (x_type, *x[1:]) if x_type is not None else None

    return tuple_mapping


class PostIO(object):
    """Post-IO processing methods.

    `PostIO` can be used to:
        (1) *drop* chunks with overlong spans;
        (2) *drop* or *merge* chunks, attributes or relations of specific types;
        (3) *absorb* specific type of attributes into chunk-type, or *exclude* it;
        (4) *infer* relations given specific grouping relation-types.

    Notes
    -----
    All methods make a **shallow copy** of input `data` before processing.
    Hence, do not use inplace modification like ``entry['chunks'].append`` or ``entry['chunks'].extend`` in the code.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.attr_sep = "♦️"

    def map_chunks(
        self,
        data: List[dict],
        chunk_type_mapping: Union[Callable, dict] = None,
        max_span_size: int = None,
    ):
        data = [{k: v for k, v in entry.items()} for entry in data]
        chunk_mapping = _make_tuple_mapping(
            chunk_type_mapping,
            aux_check=(lambda ck: ck[2] - ck[1] <= max_span_size)
            if max_span_size is not None
            else None,
        )

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Chunk mapping"
        ):
            if "chunks" in entry:
                entry["chunks"] = [
                    chunk_mapping(ck)
                    for ck in entry["chunks"]
                    if chunk_mapping(ck) is not None
                ]
            if "attributes" in entry:
                entry["attributes"] = [
                    (attr_type, chunk_mapping(ck))
                    for attr_type, ck in entry["attributes"]
                    if chunk_mapping(ck) is not None
                ]
            if "relations" in entry:
                entry["relations"] = [
                    (rel_type, chunk_mapping(head), chunk_mapping(tail))
                    for rel_type, head, tail in entry["relations"]
                    if chunk_mapping(head) is not None
                    and chunk_mapping(tail) is not None
                ]
        return data

    def map_attributes(
        self, data: List[dict], attribute_type_mapping: Union[Callable, dict] = None
    ):
        data = [{k: v for k, v in entry.items()} for entry in data]
        attribute_mapping = _make_tuple_mapping(attribute_type_mapping)

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Attribute mapping"
        ):
            if "attributes" in entry:
                entry["attributes"] = [
                    attribute_mapping(attr)
                    for attr in entry["attributes"]
                    if attribute_mapping(attr) is not None
                ]
        return data

    def map_relations(
        self, data: List[dict], relation_type_mapping: Union[Callable, dict] = None
    ):
        data = [{k: v for k, v in entry.items()} for entry in data]
        relation_mapping = _make_tuple_mapping(relation_type_mapping)

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Relation mapping"
        ):
            if "relations" in entry:
                entry["relations"] = [
                    relation_mapping(rel)
                    for rel in entry["relations"]
                    if relation_mapping(rel) is not None
                ]
        return data

    def map(self, data: List[dict], **kwargs):
        data = self.map_chunks(
            data,
            **{
                kw: kwargs.get(kw, None)
                for kw in ["chunk_type_mapping", "max_span_size"]
            }
        )
        data = self.map_attributes(
            data, **{kw: kwargs.get(kw, None) for kw in ["attribute_type_mapping"]}
        )
        data = self.map_relations(
            data, **{kw: kwargs.get(kw, None) for kw in ["relation_type_mapping"]}
        )
        return data

    def absorb_attributes(self, data: List[dict], absorb_attr_types: List[str]):
        data = [{k: v for k, v in entry.items()} for entry in data]

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Attribute absorbing"
        ):
            chunk2attrs = {ck: [] for ck in entry["chunks"]}
            for attr_type, chunk in entry["attributes"]:
                if attr_type in absorb_attr_types:
                    chunk2attrs[chunk].append(attr_type)

            chunk2new_chunk = {
                ck: (self.attr_sep.join((ck[0], *sorted(chunk2attrs[ck]))), *ck[1:])
                for ck in entry["chunks"]
            }
            entry["chunks"] = [chunk2new_chunk[ck] for ck in entry["chunks"]]
            entry["attributes"] = [
                (attr_type, chunk2new_chunk[ck])
                for attr_type, ck in entry["attributes"]
                if attr_type not in absorb_attr_types
            ]
            if "relations" in entry:
                entry["relations"] = [
                    (rel_type, chunk2new_chunk[head], chunk2new_chunk[tail])
                    for rel_type, head, tail in entry["relations"]
                ]
        return data

    def exclude_attributes(self, data: List[dict]):
        data = [{k: v for k, v in entry.items()} for entry in data]

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Attribute excluding"
        ):
            new_attributes = []
            for ck in entry["chunks"]:
                for attr_type in ck[0].split(self.attr_sep)[1:]:
                    new_attributes.append((attr_type, ck))

            chunk2new_chunk = {
                ck: (ck[0].split(self.attr_sep)[0], *ck[1:]) for ck in entry["chunks"]
            }
            entry["chunks"] = [chunk2new_chunk[ck] for ck in entry["chunks"]]
            entry["attributes"] = [
                (attr_type, chunk2new_chunk[ck])
                for attr_type, ck in entry["attributes"] + new_attributes
            ]
            if "relations" in entry:
                entry["relations"] = [
                    (rel_type, chunk2new_chunk[head], chunk2new_chunk[tail])
                    for rel_type, head, tail in entry["relations"]
                ]
        return data

    def _build_chunk2group(self, entry: dict, group_rel_types: List[str]):
        chunk2group = {ck: None for ck in entry["chunks"]}
        for rel_type, head, tail in entry["relations"]:
            if rel_type in group_rel_types:
                if chunk2group[head] is None and chunk2group[tail] is None:
                    new_group = set([head, tail])
                    chunk2group[head] = new_group
                    chunk2group[tail] = new_group
                elif chunk2group[head] is None and chunk2group[tail] is not None:
                    chunk2group[tail].add(head)
                    chunk2group[head] = chunk2group[tail]
                elif chunk2group[head] is not None and chunk2group[tail] is None:
                    chunk2group[head].add(tail)
                    chunk2group[tail] = chunk2group[head]
                elif chunk2group[head] != chunk2group[tail]:
                    assert len(chunk2group[head] & chunk2group[tail]) == 0
                    union_group = chunk2group[head] | chunk2group[tail]
                    for ck in union_group:
                        chunk2group[ck] = union_group
        return chunk2group

    def _detect_relations(
        self, entry: dict, group_rel_types: List[str], chunk2group: dict
    ):
        new_relations = []
        for rel_type, head, tail in entry["relations"]:
            if rel_type not in group_rel_types:
                if chunk2group[head] is None and chunk2group[tail] is None:
                    continue
                elif chunk2group[head] is None and chunk2group[tail] is not None:
                    curr_new_relations = [
                        (rel_type, head, ck) for ck in chunk2group[tail]
                    ]
                elif chunk2group[head] is not None and chunk2group[tail] is None:
                    curr_new_relations = [
                        (rel_type, ck, tail) for ck in chunk2group[head]
                    ]
                else:
                    assert len(chunk2group[head] & chunk2group[tail]) == 0
                    curr_new_relations = [
                        (rel_type, hck, tck)
                        for hck in chunk2group[head]
                        for tck in chunk2group[tail]
                    ]
                curr_new_relations = [
                    rel
                    for rel in curr_new_relations
                    if rel not in entry["relations"] and rel not in new_relations
                ]
                new_relations.extend(curr_new_relations)
        return new_relations

    def infer_relations(self, data: List[dict], group_rel_types: List[str]):
        data = [{k: v for k, v in entry.items()} for entry in data]

        for entry in tqdm.tqdm(
            data, disable=not self.verbose, ncols=100, desc="Relation inferring"
        ):
            chunk2group = self._build_chunk2group(entry, group_rel_types)
            new_relations = self._detect_relations(entry, group_rel_types, chunk2group)
            entry["relations"] = entry["relations"] + new_relations
        return data
