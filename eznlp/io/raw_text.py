# -*- coding: utf-8 -*-
import logging
import tqdm

from .base import IO
from ..utils.segmentation import segment_text_uniformly

logger = logging.getLogger(__name__)


class RawTextIO(IO):
    """An IO interface of raw text files. 
    """
    def __init__(self, tokenize_callback=None, max_len: int=None, encoding=None, verbose: bool=True):
        super().__init__(is_tokenized=False, tokenize_callback=tokenize_callback, encoding=encoding, verbose=verbose)
        
        assert not (tokenize_callback is not None and max_len is None)
        self.max_len = max_len
        self.document_seperator = "-DOCSTART-"
        
        
    def read(self, file_path):
        with open(file_path, 'rb') as f:
            byte_lines = [line for line in f if len(line.rstrip()) > 0]
        
        data = []
        if self.max_len is None:
            for byte_line in tqdm.tqdm(byte_lines, disable=not self.verbose, ncols=100, desc="Loading raw text data"):
                line = byte_line.decode(self.encoding)
                # `tokenize_callback` must be None
                data.append(line)
        
        else:
            tokenized_doc = []
            for byte_line in tqdm.tqdm(byte_lines, disable=not self.verbose, ncols=100, desc="Loading raw text data"):
                line = byte_line.decode(self.encoding)
                
                if line.startswith(self.document_seperator):
                    if len(tokenized_doc) > 0:
                        for start, end in segment_text_uniformly(tokenized_doc, max_span_size=self.max_len):
                            data.append(" ".join(tokenized_doc[start:end]))
                    tokenized_doc = []
                    
                elif self.tokenize_callback is None:
                    tokenized_doc.extend(line.split(" "))
                else:
                    tokenized_doc.extend(self.tokenize_callback(line))
            
            if len(tokenized_doc) > 0:
                for start, end in segment_text_uniformly(tokenized_doc, max_span_size=self.max_len):
                    data.append(" ".join(tokenized_doc[start:end]))
        
        return data
        
        
    def write(self, data, file_path):
        with open(file_path, 'w', encoding=self.encoding) as f:
            for line in data:
                f.write(line)
                f.write("\n")
