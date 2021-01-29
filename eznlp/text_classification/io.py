# -*- coding: utf-8 -*-
from ..data import TokenSequence


class TabularIO(object):
    """
    An IO interface of Tabular-format files. 
    
    """
    def __init__(self, text_col_id=0, label_col_id=1, tokenize_callback=None, **kwargs):
        self.text_col_id = text_col_id
        self.label_col_id = label_col_id
        self.tokenize_callback = tokenize_callback
        self.kwargs = kwargs
        
    def read(self, file_path, encoding=None, sep=None, sentence_sep=None):
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                
                line_seperated = line.split(sep)
                raw_text = line_seperated[self.text_col_id].strip()
                if sentence_sep is not None:
                    raw_text = raw_text.replace(sentence_sep, "\n")
                
                tokens = TokenSequence.from_raw_text(raw_text, self.tokenize_callback, **self.kwargs)
                label = line_seperated[self.label_col_id].strip()
                data.append({'raw_text': raw_text, 'tokens': tokens, 'label': label})
                
        return data
    