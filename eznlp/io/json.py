# -*- coding: utf-8 -*-
import json

from ..token import TokenSequence
from .base import IO


class JsonIO(IO):
    """
    An IO Interface of Json files. 
    
    """
    def __init__(self, 
                 text_key='tokens', 
                 chunk_key='entities', 
                 chunk_type_key='type', 
                 chunk_start_key='start', 
                 chunk_end_key='end', 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.text_key = text_key
        self.chunk_key = chunk_key
        self.chunk_type_key = chunk_type_key
        self.chunk_start_key = chunk_start_key
        self.chunk_end_key = chunk_end_key
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
    def read(self, file_path):
        with open(file_path, 'r', encoding=self.encoding) as f:
            raw_data = json.load(f)
        
        data = []
        for raw_entry in raw_data:
            tokens = TokenSequence.from_tokenized_text(raw_entry[self.text_key], **self.kwargs)
            chunks = [(chunk[self.chunk_type_key], 
                       chunk[self.chunk_start_key],
                       chunk[self.chunk_end_key]) for chunk in raw_entry[self.chunk_key]]
            data.append({'tokens': tokens, 'chunks': chunks})
            
        return data

