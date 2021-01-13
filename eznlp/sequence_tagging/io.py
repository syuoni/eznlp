# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd

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
    def __init__(self, text_col_id=0, tag_col_id=1, scheme='BIO1', additional_col_id2name=None, 
                 line_sep_starts=None, breaking_for_types=True, **kwargs):
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
        self.kwargs = kwargs
        
        
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
                        tokens = TokenSequence.from_tokenized_text(text, additional_tags, **self.kwargs)
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
        


class BratIO(object):
    """
    An IO interface of brat-format files. 
    """
    def __init__(self, attr_names=None, pre_inserted_spaces=True, tokenize_callback=None, **kwargs):
        self.attr_names = [] if attr_names is None else attr_names
        self.pre_inserted_spaces = pre_inserted_spaces
        self.tokenize_callback = tokenize_callback
        self.kwargs = kwargs
        self.line_sep = "\r\n"
        self.attr_sep = "<a>"
        
    def _parse_chunk_ann(self, ann):
        chunk_id, chunk_type_pos, chunk_text = ann.rstrip(self.line_sep).split("\t")
        chunk_type, chunk_start_in_text, chunk_end_in_text = chunk_type_pos.split(" ")
        chunk_start_in_text = int(chunk_start_in_text)
        chunk_end_in_text = int(chunk_end_in_text)
        return chunk_id, (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        
    def _build_chunk_ann(self, chunk_id, text_chunk):
        chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
        return f"{chunk_id}\t{chunk_type} {chunk_start_in_text} {chunk_end_in_text}\t{chunk_text}"
    
    def _parse_attr_ann(self, ann):
        attr_id, attr_name_and_chunk_id = ann.rstrip(self.line_sep).split("\t")
        attr_name, chunk_id = attr_name_and_chunk_id.split(" ")
        return attr_id, (attr_name, chunk_id)
    
    def _build_attr_ann(self, attr_id, attr_name_and_chunk_id):
        attr_name, chunk_id = attr_name_and_chunk_id
        return f"{attr_id}\t{attr_name} {chunk_id}"
    
    
    def read(self, file_path, encoding=None):
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
        with open(file_path.replace('.txt', '.ann'), 'r', encoding=encoding) as f:
            anns = f.readlines()
            assert all(ann.startswith(('T', 'A', 'R')) for ann in anns)
            
        if ("\n" in text) and (self.line_sep not in text):
            text = text.replace("\n", self.line_sep)
            
        # Parse chunks
        text_chunks = dict([self._parse_chunk_ann(ann) for ann in anns if ann.startswith('T')])
        if len(text_chunks) == 0:
            return []
        
        if self.pre_inserted_spaces:
            text, text_chunks = self._remove_pre_inserted_spaces(text, text_chunks)
            
        self._check_text_chunks(text, text_chunks)
        
        # Build dataframe
        df = pd.DataFrame(text_chunks, index=['text', 'type', 'start_in_text', 'end_in_text']).T
        for attr_name in self.attr_names:
            df[attr_name] = 'F'
            
        # Parse chunk attributes
        text_attrs = dict([self._parse_attr_ann(ann) for ann in anns if ann.startswith('A')])
        for attr_id, (attr_name, chunk_id) in text_attrs.items():
            if attr_name in self.attr_names:
                df.loc[chunk_id, attr_name] = 'T'
            
        for attr_name in self.attr_names:
            df['type'] = df['type'].str.cat(df[attr_name], sep=self.attr_sep)
        
        data = []
        line_start = 0
        for line_cut in re.finditer(self.line_sep, text):
            line_end = line_cut.start()
            
            if line_end > line_start:
                data.append(self._build_entry(text, df, line_start, line_end))
            line_start = line_cut.end()
            
        if len(text) > line_start:
            data.append(self._build_entry(text, df, line_start))
        return data
    
    
    def _build_entry(self, text: str, df: pd.DataFrame, line_start=None, line_end=None):
        line_start = 0 if line_start is None else line_start
        line_end = len(text) if line_end is None else line_end
        
        curr_df = df.loc[(line_start <= df['start_in_text']) & (df['end_in_text'] <= line_end)]
        chunks = []
        for chunk_id, (chunk_type, chunk_start_in_text, chunk_end_in_text) in curr_df[['type', 'start_in_text', 'end_in_text']].iterrows():
            chunk_start = chunk_start_in_text - line_start
            chunk_end = chunk_end_in_text - line_start
            chunks.append((chunk_type, chunk_start, chunk_end))
            
        return {'tokens': TokenSequence.from_tokenized_text(list(text[line_start:line_end]), **self.kwargs),
                'chunks': chunks}
        
    def _remove_pre_inserted_spaces(self, text: str, text_chunks: dict):
        """
        Chinese text may be tokenized and re-joined by spaces before annotation. 
        """
        is_inserted = [int(c == " ") for c in text.replace("  ", " #")]
        num_inserted = np.cumsum(is_inserted).tolist()
        # The positions of exact pre-inserted spaces should be mapped to positions NEXT to them
        num_inserted = [n - i for n, i in zip(num_inserted, is_inserted)]
        num_inserted.append(num_inserted[-1])
        
        text = text.replace("  ", " <#>").replace(" ", "").replace("<#>", " ")
        
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            chunk_text = chunk_text.replace("  ", " <#>").replace(" ", "").replace("<#>", " ")
            chunk_start_in_text -= num_inserted[chunk_start_in_text]
            chunk_end_in_text -= num_inserted[chunk_end_in_text]
            
            text_chunks[chunk_id] = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            
        return text, text_chunks
        
    
    def _check_text_chunks(self, text: str, text_chunks: dict):
        # Check chunk text consistenty
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            assert text[chunk_start_in_text:chunk_end_in_text] == chunk_text
            
            
    def write(self, data, file_path, encoding=None):
        text_lines = []
        text_chunks = {}
        
        chunk_idx, attr_idx = 1, 1
        chunk_lines, attr_lines = [], []
        
        line_start = 0
        for curr_data in data:
            line_text = "".join(curr_data['tokens'].raw_text)
            text_lines.append(line_text)
            
            for chunk_type, chunk_start, chunk_end in curr_data['chunks']:
                chunk_text = line_text[chunk_start:chunk_end]
                chunk_start_in_text = chunk_start + line_start
                chunk_end_in_text = chunk_end + line_start
                
                text_chunk = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
                text_chunks[f"T{chunk_idx}"] = text_chunk
                chunk_idx += 1
                
            line_start = line_start + len(line_text) + len(self.line_sep)
            
        text = self.line_sep.join(text_lines)
        self._check_text_chunks(text, text_chunks)
        
        if self.pre_inserted_spaces:
            text, text_chunks = self._insert_spaces(text, text_chunks)
            
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            
            chunk_type, *attr_values = chunk_type.split(self.attr_sep)
            text_chunk = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            chunk_lines.append(self._build_chunk_ann(chunk_id, text_chunk))
            
            for attr_name, attr_v in zip(self.attr_names, attr_values):
                if attr_v == 'T':
                    attr_lines.append(self._build_attr_ann(f"A{attr_idx}", (attr_name, chunk_id)))
                    attr_idx += 1
                    
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(text.replace(self.line_sep, "\n"))
            f.write("\n")
        with open(file_path.replace('.txt', '.ann'), 'w', encoding=encoding) as f:
            f.write("\n".join(chunk_lines))
            f.write("\n")
            f.write("\n".join(attr_lines))
            f.write("\n")
            
            
    def _tokenize_and_rejoin(self, text: str):
        text = " ".join(self.tokenize_callback(text))
        
        text = text.replace("“ ", "“").replace(" ”", "”")
        text = text.replace(" ／ ", "／")
        text = text.replace(" ：", "：")
        return text
    
    
    def _insert_spaces(self, text: str, text_chunks: dict):
        """
        Chinese text may be tokenized and re-joined by spaces before annotation. 
        """
        ori_num_chars = len(text)
        text = self.line_sep.join([self._tokenize_and_rejoin(line) for line in text.split(self.line_sep)])
        
        is_inserted = [int(c == " ") for c in text.replace("  ", " #")]
        num_inserted = np.cumsum(is_inserted).tolist()
        num_inserted = [n for n, i in zip(num_inserted, is_inserted) if i == 0]
        assert len(num_inserted) == ori_num_chars
        num_inserted.append(num_inserted[-1])
        
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            chunk_start_in_text += num_inserted[chunk_start_in_text]
            chunk_end_in_text += num_inserted[chunk_end_in_text] - int(num_inserted[chunk_end_in_text] > num_inserted[chunk_end_in_text-1])
            assert text[chunk_start_in_text:chunk_end_in_text].replace(" ", "") == chunk_text.replace(" ", "")
            chunk_text = text[chunk_start_in_text:chunk_end_in_text]
            
            text_chunks[chunk_id] = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            
        return text, text_chunks
        
        