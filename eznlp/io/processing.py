# -*- coding: utf-8 -*-
from typing import Union, List, Callable


def _make_tuple_mapping(type_mapping: Union[Callable, dict]=None):
    def tuple_mapping(x):
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
        (2) *drop* or *merge* chunks, attributes or relations of specific types. 
    """
    def __init__(self, 
                 max_span_size: int=None, 
                 chunk_type_mapping: Union[Callable, dict]=None, 
                 attribute_type_mapping: Union[Callable, dict]=None, 
                 relation_type_mapping: Union[Callable, dict]=None):
        self.max_span_size = max_span_size
        self.chunk_mapping = _make_tuple_mapping(chunk_type_mapping)
        self.attribute_mapping = _make_tuple_mapping(attribute_type_mapping)
        self.relation_mapping = _make_tuple_mapping(relation_type_mapping)
        
    def _map_chunk(self, chunk: tuple):
        chunk_type, start, end = chunk
        if self.max_span_size is not None and end - start > self.max_span_size:
            return None
        else:
            return self.chunk_mapping(chunk)
        
    def _map_attribute(self, attribute: tuple):
        attr_type, chunk = attribute
        new_chunk = self._map_chunk(chunk)
        if new_chunk is None:
            return None
        else:
            return self.attribute_mapping((attr_type, new_chunk))
        
    def _map_relation(self, relation: tuple):
        rel_type, head, tail = relation
        new_head = self._map_chunk(head)
        new_tail = self._map_chunk(tail)
        if new_head is None or new_tail is None:
            return None
        else:
            return self.relation_mapping((rel_type, new_head, new_tail))
        
    def map_process(self, data: List[dict]):
        for entry in data:
            if 'chunks' in entry:
                entry['chunks'] = [self._map_chunk(ck) for ck in entry['chunks'] if self._map_chunk(ck) is not None]
            if 'attributes' in entry:
                entry['attributes'] = [self._map_attribute(attr) for attr in entry['attributes'] if self._map_attribute(attr) is not None]
            if 'relations' in entry:
                entry['relations'] = [self._map_relation(rel) for rel in entry['relations'] if self._map_relation(rel) is not None]
        return data
        
        
    def absorb_attributes(self, data: List[dict]):
        pass
        
        
    def process(self, data: List[dict]):
        data = self.map_process(data)
        return data
