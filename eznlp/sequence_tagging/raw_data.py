# -*- coding: utf-8 -*-
from ..token import TokenSequence
from .transition import SchemeTranslator, ChunksTagsTranslator


def _build_data_entry(example, columns, text_col, trg_col, attach_additional_tags, **kwargs):
    if len(example[0]) == 0:
        return None
    
    tokenized_text = example[columns.index(text_col)]
    tags = example[columns.index(trg_col)]
    if attach_additional_tags:
        additional_tags = {c: ex_part for c, ex_part in zip(columns, example) if c not in (text_col, trg_col)}
    else:
        additional_tags = None
    
    tokens = TokenSequence.from_tokenized_text(tokenized_text, additional_tags=additional_tags, **kwargs)
    return {'tokens': tokens, 'tags': tags}



def parse_conll_file(file_path, encoding=None, sep=' ', raw_scheme='BIO1', scheme='BIOES', 
                     columns=['text', 'pos_tag', 'chunking_tag'], text_col='text', trg_col='chunking_tag', 
                     attach_additional_tags=False, skip_docstart=True, max_examples=None, **kwargs):
    scheme_translator = SchemeTranslator(from_scheme=raw_scheme, to_scheme=scheme)
    
    data = []
    example = [[] for c in columns]
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            
            if skip_docstart and line.startswith("-DOCSTART-"):
                continue
            
            if line == '':
                curr_data = _build_data_entry(example, columns, text_col, trg_col, attach_additional_tags, **kwargs)
                if curr_data is not None:
                    curr_data['tags'] = scheme_translator.translate(curr_data['tags'])
                    data.append(curr_data)
                example = [[] for c in columns]
                
                if max_examples is not None and len(data) >= max_examples:
                    break
            else:
                for ex_part, ex_part_to_append in zip(example, line.split(sep)):
                    ex_part.append(ex_part_to_append)
                    
        curr_data = _build_data_entry(example, columns, text_col, trg_col, attach_additional_tags, **kwargs)
        if curr_data is not None:
            data.append(curr_data)
            
    return data




class ConllReader(object):
    """
    A reader of CoNLL-format files. 
    
    Parameters
    ----------
    line_sep_starts: list of str
        For Conll2003, `line_sep_starts` should be `["-DOCSTART-"]`
        For OntoNotes 5, `line_sep_starts` should be `["#begin", "#end", "pt/"]`
    """
    def __init__(self, text_col_id=0, tag_col_id=1, scheme='BIO1', additional_col_id2name=None, line_sep_starts=None, breaking_for_types=True):
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
            assert all(isinstance(start) for start in line_sep_starts)
            self.line_sep_starts = line_sep_starts
            
        self.breaking_for_types = breaking_for_types
            
            
    def read(self, file_path, encoding=None, sep=" "):
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            text, tags = [], []
            additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
            
            for line in f:
                line = line.strip()
                
                if self._is_line_seperator(line):
                    if len(text) == 0:
                        continue
                    additional_tags = {self.additional_col_id2name[col_id]: atags for col_id, atags in additional.items()}
                    tokens = TokenSequence.from_tokenized_text(text, additional_tags)
                    
                    chunks = self.translator.tags2chunks(tags, self.breaking_for_types)
                    data.append([tokens, chunks])
                    
                    text, tags = [], []
                    additional = {col_id: [] for col_id in self.additional_col_id2name.keys()}
                else:
                    line_seperated = line.split(sep)
                    text.append(line_seperated[self.text_col_id])
                    tags.append(line_seperated[self.tag_col_id])
                    for col_id in self.additional_col_id2name.keys():
                        additional[col_id].append(line_seperated[col_id])
                        
        return data
    
    
    def _is_line_seperator(self, line: str):
        if line.strip() == "":
            return True
        
        for start in self.line_sep_starts:
            if line.startswith(start):
                return True
            
        return False
        

