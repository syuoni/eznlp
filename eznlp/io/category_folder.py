# -*- coding: utf-8 -*-
from typing import List
import tqdm
import glob

from ..token import TokenSequence
from .base import IO


class CategoryFolderIO(IO):
    """
    An IO interface of text files in category-specific folders. 
    
    """
    def __init__(self, 
                 categories: List[str], 
                 mapping=None, 
                 tokenize_callback=None, 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.categories = categories
        self.mapping = {} if mapping is None else mapping
        self.tokenize_callback = tokenize_callback
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
    def read(self, folder_path):
        data = []
        for label in self.categories:
            file_paths = glob.glob(f"{folder_path}/{label}/*.txt")
            for path in tqdm.tqdm(file_paths, disable=not self.verbose, ncols=100, desc=f"Loading data in folder {label}"):
                with open(path, encoding=self.encoding) as f:
                    raw_text = f.read()
                for pattern, repl in self.mapping.items():
                    raw_text = raw_text.replace(pattern, repl)
                    
                tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
                data.append({'tokens': tokens, 'label': label})
                
        return data
    
    