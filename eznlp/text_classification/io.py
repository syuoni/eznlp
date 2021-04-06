# -*- coding: utf-8 -*-
from typing import List
import tqdm
import glob
import pandas

from ..token import TokenSequence


class TabularIO(object):
    """
    An IO interface of Tabular-format files. 
    
    """
    def __init__(self, text_col_id=0, label_col_id=1, mapping=None, tokenize_callback=None, verbose=True, **kwargs):
        self.text_col_id = text_col_id
        self.label_col_id = label_col_id
        self.mapping = {} if mapping is None else mapping
        self.tokenize_callback = tokenize_callback
        self.verbose = verbose
        self.kwargs = kwargs
        
    def read(self, file_path, encoding=None, sep=None):
        df = pandas.read_csv(file_path, header=None, dtype=str, encoding=encoding, sep=sep, engine='python')
        
        data = []
        for _, line in tqdm.tqdm(df.iterrows(), total=df.shape[0], disable=not self.verbose, ncols=100, desc="Loading tabular data"):
            raw_text, label = line.iloc[self.text_col_id].strip(), line.iloc[self.label_col_id].strip()
            for pattern, repl in self.mapping.items():
                raw_text = raw_text.replace(pattern, repl)
                
            tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
            data.append({'tokens': tokens, 'label': label})
            
        return data
    
    
class FolderIO(object):
    """
    An IO interface of text files in category-folders
    """
    def __init__(self, categories: List[str], mapping=None, tokenize_callback=None, verbose=True, **kwargs):
        self.categories = categories
        self.mapping = {} if mapping is None else mapping
        self.tokenize_callback = tokenize_callback
        self.verbose = verbose
        self.kwargs = kwargs
    
    def read(self, folder_path, encoding=None):
        data = []
        for label in self.categories:
            file_paths = glob.glob(f"{folder_path}/{label}/*.txt")
            for path in tqdm.tqdm(file_paths, disable=not self.verbose, ncols=100, desc=f"Loading data in folder {label}"):
                with open(path, encoding=encoding) as f:
                    raw_text = f.read()
                for pattern, repl in self.mapping.items():
                    raw_text = raw_text.replace(pattern, repl)
                    
                tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
                data.append({'tokens': tokens, 'label': label})
                
        return data
    
    