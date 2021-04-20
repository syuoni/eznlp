# -*- coding: utf-8 -*-
import tqdm
import pandas

from ..token import TokenSequence
from .base import IO


class TabularIO(IO):
    """
    An IO interface of Tabular-format files. 
    
    """
    def __init__(self, 
                 text_col_id=0, 
                 label_col_id=1, 
                 sep=',', 
                 header=None, 
                 mapping=None, 
                 tokenize_callback=None, 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.text_col_id = text_col_id
        self.label_col_id = label_col_id
        self.sep = sep
        self.engine = 'python' if (len(sep) > 1 and sep != '\s+') else 'c'
        self.header = header
        self.mapping = {} if mapping is None else mapping
        self.tokenize_callback = tokenize_callback
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
    def read(self, file_path):
        df = pandas.read_csv(file_path, encoding=self.encoding, sep=self.sep, header=self.header, dtype=str, na_filter=False, engine=self.engine)
        
        data = []
        for _, line in tqdm.tqdm(df.iterrows(), total=df.shape[0], disable=not self.verbose, ncols=100, desc="Loading tabular data"):
            raw_text, label = line.iloc[self.text_col_id].strip(), line.iloc[self.label_col_id].strip()
            for pattern, repl in self.mapping.items():
                raw_text = raw_text.replace(pattern, repl)
                
            tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
            data.append({'tokens': tokens, 'label': label})
            
        return data

