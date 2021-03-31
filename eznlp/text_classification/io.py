# -*- coding: utf-8 -*-
import tqdm
import pandas

from ..token import TokenSequence


class TabularIO(object):
    """
    An IO interface of Tabular-format files. 
    
    """
    def __init__(self, text_col_id=0, label_col_id=1, mapping=None, tokenize_callback=None, **kwargs):
        self.text_col_id = text_col_id
        self.label_col_id = label_col_id
        self.mapping = {} if mapping is None else mapping
        self.tokenize_callback = tokenize_callback
        self.kwargs = kwargs
        
    def read(self, file_path, encoding=None, sep=None):
        df = pandas.read_csv(file_path, header=None, dtype=str, encoding=encoding, sep=sep, engine='python')
        
        data = []
        for _, line in tqdm.tqdm(df.iterrows(), total=df.shape[0], disable=df.shape[0]<10_000, desc="loading tabular data"):
            raw_text, label = line.iloc[self.text_col_id].strip(), line.iloc[self.label_col_id].strip()
            for pattern, repl in self.mapping.items():
                raw_text = raw_text.replace(pattern, repl)
                
            tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
            data.append({'tokens': tokens, 'label': label})
            
        return data
    