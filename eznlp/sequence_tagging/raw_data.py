# -*- coding: utf-8 -*-
import re

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
    def __init__(self, use_attrs=None):
        self.use_attrs = use_attrs
        
    
    def read(self, file_path, encoding=None):
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        with open(file_path.replace('.txt', '.ann'), 'r', encoding=encoding) as f:
            anns = f.readlines()
            
        if ("\n" in text) and ("\r\n" not in text):
            text = text.replace("\n", "\r\n")
            
        text_chunks = dict([self._parse_chunk_ann(ann) for ann in anns if ann.startswith('T')])
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            assert text[chunk_start_in_text:chunk_end_in_text].strip() == chunk_text.strip()
            
        # Update chunk_type for potential attributes
        if self.use_attrs is not None:
            attrs = [self._parse_attr_ann(ann) for ann in anns if ann.startswith('A')]
            for chunk_id, attr_type in attrs:
                if attr_type not in self.use_attrs:
                    continue
                
                chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunks[chunk_id]
                ori_type, *chunk_attr_types = chunk_type.split('<s>')
                chunk_attr_types.append(attr_type)
                chunk_type = '<s>'.join([ori_type] + sorted(chunk_attr_types))
                text_chunks[chunk_id] = chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text
                
        text_chunks = list(text_chunks.values())
        
        data = []
        line_start = 0
        for line_cut in re.finditer("\r\n", text):
            line_end = line_cut.start()
            if line_end > line_start:
                data.append(self._build_entry(text, text_chunks, line_start, line_end))
            line_start = line_cut.end()
            
        if len(text) > line_start:
            data.append(self._build_entry(text, text_chunks, line_start))
        return data
            
        
    def _build_entry(self, text: str, text_chunks: list, line_start=None, line_end=None):
        line_start = 0 if line_start is None else line_start
        line_end = len(text) if line_end is None else line_end
        
        chunks = []
        for chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text in text_chunks:
            if (line_start <= chunk_start_in_text) and (chunk_end_in_text <= line_end):
                # TODO: chunk_start != chunk_start_in_text
                chunk_start = chunk_start_in_text - line_start
                chunk_end = chunk_end_in_text - line_start
                chunks.append((chunk_type, chunk_start, chunk_end))
                
        return {'tokens': TokenSequence.from_tokenized_text(list(text[line_start:line_end])),
                'chunks': chunks}
        
        
    def _parse_chunk_ann(self, ann):
        chunk_id, chunk_type_pos, chunk_text = ann.strip().split("\t")
        chunk_type, chunk_start_in_text, chunk_end_in_text = chunk_type_pos.split(" ")
        chunk_start_in_text = int(chunk_start_in_text)
        chunk_end_in_text = int(chunk_end_in_text)
        return chunk_id, (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        
    def _parse_attr_ann(self, ann):
        attr_id, attr_type_and_chunk_id = ann.strip().split("\t")
        attr_type, chunk_id = attr_type_and_chunk_id.split(" ")
        return chunk_id, attr_type
    
