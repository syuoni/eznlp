# -*- coding: utf-8 -*-
from ..token import TokenSequence
from ..utils import ChunksTagsTranslator
from .base import IO


class ConllIO(IO):
    """
    An IO interface of CoNLL-format files. 
    
    Parameters
    ----------
    line_sep_starts: list of str
        For Conll2003, `line_sep_starts` should be `["-DOCSTART-"]`
        For OntoNotes5 (i.e., Conll2012), `line_sep_starts` should be `["#begin", "#end", "pt/"]`
    """
    def __init__(self, 
                 text_col_id=0, 
                 tag_col_id=1, 
                 sep=None, 
                 scheme='BIO1', 
                 additional_col_id2name=None, 
                 line_sep_starts=None, 
                 breaking_for_types=True, 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.text_col_id = text_col_id
        self.tag_col_id = tag_col_id
        self.sep = sep
        
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
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
    def read(self, file_path):
        data = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            text, tags = [], []
            additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
            
            for line in f:
                line = line.strip()
                
                if self._is_line_seperator(line):
                    if len(text) > 0:
                        additional_tags = {self.additional_col_id2name[col_id]: atags for col_id, atags in additional.items()}
                        tokens = TokenSequence.from_tokenized_text(text, additional_tags, **self.kwargs)
                        chunks = self.translator.tags2chunks(tags, self.breaking_for_types)
                        data.append({'tokens': tokens, 'chunks': chunks})
                        
                        text, tags = [], []
                        additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
                else:
                    line_seperated = line.split(self.sep)
                    text.append(line_seperated[self.text_col_id])
                    tags.append(line_seperated[self.tag_col_id])
                    for col_id in self.additional_col_id2name.keys():
                        additional[col_id].append(line_seperated[col_id])
                        
            if len(text) > 0:
                additional_tags = {self.additional_col_id2name[col_id]: atags for col_id, atags in additional.items()}
                tokens = TokenSequence.from_tokenized_text(text, additional_tags, **self.kwargs)
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

