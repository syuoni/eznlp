# -*- coding: utf-8 -*-
import re

from ..data import TokenSequence
from .transition import ChunksTagsTranslator


class ConllIO(object):
    """
    An IO interface of CoNLL-format files. 
    
    Parameters
    ----------
    line_sep_starts: list of str
        For Conll2003, `line_sep_starts` should be `["-DOCSTART-"]`
        For OntoNotes5 (i.e., Conll2012), `line_sep_starts` should be `["#begin", "#end", "pt/"]`
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
        


class BratIO(object):
    """
    An IO interface of brat-format files. 
    """
    def __init__(self, use_attrs=None):
        self.use_attrs = use_attrs
        self.type_sep = "<s>"
        self.line_sep = "\r\n"
        
    def write(self, data, file_path, encoding=None):
        text_lines = []
        chunk_lines, attr_lines = [], []
        
        chunk_idx, attr_idx = 1, 1
        line_start = 0
        for curr_data in data:
            line_text = "".join(curr_data['tokens'].raw_text)
            text_lines.append(line_text)
            
            for chunk_type, chunk_start, chunk_end in curr_data['chunks']:
                ori_chunk_type, *chunk_attr_types = chunk_type.split(self.type_sep)
                chunk_text = line_text[chunk_start:chunk_end]
                chunk_start_in_text = chunk_start + line_start
                chunk_end_in_text = chunk_end + line_start
                text_chunk = (chunk_text, ori_chunk_type, chunk_start_in_text, chunk_end_in_text)
                chunk_lines.append(self._build_chunk_ann(f"T{chunk_idx}", text_chunk))
                
                for attr_type in chunk_attr_types:
                    attr = (attr_type, f"T{chunk_idx}")
                    attr_lines.append(self._build_attr_ann(f"A{attr_idx}", attr))
                    attr_idx += 1
                    
                chunk_idx += 1
                
            line_start = line_start + len(line_text) + len(self.line_sep)
            
        with open(file_path, 'w', encoding=encoding) as f:
            f.write("\n".join(text_lines))
            f.write("\n")
        with open(file_path.replace('.txt', '.ann'), 'w', encoding=encoding) as f:
            f.write("\n".join(chunk_lines))
            f.write("\n")
            f.write("\n".join(attr_lines))
            f.write("\n")
            
            
    def read(self, file_path, encoding=None):
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        with open(file_path.replace('.txt', '.ann'), 'r', encoding=encoding) as f:
            anns = f.readlines()
            
        if ("\n" in text) and (self.line_sep not in text):
            text = text.replace("\n", self.line_sep)
            
        text_chunks = dict([self._parse_chunk_ann(ann) for ann in anns if ann.startswith('T')])
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            assert text[chunk_start_in_text:chunk_end_in_text].strip() == chunk_text.strip()
            
        # Update chunk_type for potential attributes
        if self.use_attrs is not None:
            attrs = [self._parse_attr_ann(ann) for ann in anns if ann.startswith('A')]
            for attr_id, (attr_type, chunk_id) in attrs:
                if attr_type not in self.use_attrs:
                    continue
                
                chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunks[chunk_id]
                ori_chunk_type, *chunk_attr_types = chunk_type.split(self.type_sep)
                chunk_attr_types.append(attr_type)
                chunk_type = self.type_sep.join([ori_chunk_type] + sorted(chunk_attr_types))
                text_chunks[chunk_id] = chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text
                
        text_chunks = list(text_chunks.values())
        
        data = []
        line_start = 0
        for line_cut in re.finditer(self.line_sep, text):
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
        
    def _build_chunk_ann(self, chunk_id, text_chunk):
        chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
        return f"{chunk_id}\t{chunk_type} {chunk_start_in_text} {chunk_end_in_text}\t{chunk_text}"
    
    def _parse_attr_ann(self, ann):
        attr_id, attr_type_and_chunk_id = ann.strip().split("\t")
        attr_type, chunk_id = attr_type_and_chunk_id.split(" ")
        return attr_id, (attr_type, chunk_id)
    
    def _build_attr_ann(self, attr_id, attr):
        attr_type, chunk_id = attr
        return f"{attr_id}\t{attr_type} {chunk_id}"
        
