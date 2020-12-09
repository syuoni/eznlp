# -*- coding: utf-8 -*-
from ..token import TokenSequence
from .transition import ChunksTagsTranslator


class ConllReader(object):
    """
    A reader of CoNLL-format files. 
    
    Parameters
    ----------
    line_sep_starts: list of str
        For Conll2003, `line_sep_starts` should be `["-DOCSTART-"]`
        For OntoNotes 5, `line_sep_starts` should be `["#begin", "#end", "pt/"]`
    """
    def __init__(self, text_col_id=0, tag_col_id=1, scheme='BIO1', 
                 additional_col_id2name=None, line_sep_starts=None, breaking_for_types=True):
        self.text_col_id = text_col_id
        self.tag_col_id = tag_col_id
        
        self.translator = ChunksTagsTranslator(scheme=scheme)
        if additional_col_id2name is None:
            self.additional_col_id2name = {}
        else:
            assert all(isinstance(col_id, int) for col_id in additional_col_id2name.keys())
            assert all(isinstance(col_name, str) for col_name in additional_col_id2name.values())
            self.additional_col_id2name = additional_col_id2name
            
        if line_sep_starts is None:
            self.line_sep_starts = []
        else:
            assert all(isinstance(start, str) for start in line_sep_starts)
            self.line_sep_starts = line_sep_starts
            
        self.breaking_for_types = breaking_for_types
            
            
    def read(self, file_path, encoding=None, sep=None):
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            text, tags = [], []
            additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
            
            for line in f:
                line = line.strip()
                
                if self._is_line_seperator(line):
                    if len(text) > 0:
                        additional_tags = {self.additional_col_id2name[col_id]: atags for col_id, atags in additional.items()}
                        tokens = TokenSequence.from_tokenized_text(text, additional_tags)
                        chunks = self.translator.tags2chunks(tags, self.breaking_for_types)
                        data.append({'tokens': tokens, 'chunks': chunks})
                        
                        text, tags = [], []
                        additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
                else:
                    line_seperated = line.split(sep)
                    text.append(line_seperated[self.text_col_id])
                    tags.append(line_seperated[self.tag_col_id])
                    for col_id in self.additional_col_id2name.keys():
                        additional[col_id].append(line_seperated[col_id])
                        
            if len(text) > 0:
                additional_tags = {self.additional_col_id2name[col_id]: atags for col_id, atags in additional.items()}
                tokens = TokenSequence.from_tokenized_text(text, additional_tags)
                chunks = self.translator.tags2chunks(tags, self.breaking_for_types)
                data.append({'tokens': tokens, 'chunks': chunks})
            
        return data
    
    
    def _is_line_seperator(self, line: str):
        if line.strip() == "":
            return True
        
        for start in self.line_sep_starts:
            if line.startswith(start):
                return True
            
        return False
        


class BratReader(object):
    """
    A reader of brat-format files
    """
    def __init__(self):
        pass
    
    def read(self, file_path, encoding=None):
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        with open(file_path.replace('.txt', '.ann'), 'r', encoding=encoding) as f:
            ann = f.readlines()
            
        data = []
        
        
            
            
            
    
    
    
        
        
        
    
