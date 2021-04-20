# -*- coding: utf-8 -*-
import re
import numpy
import pandas

from ..token import TokenSequence
from ..utils import segment_text_with_hierarchical_seps, segment_text_uniformly
from .base import IO


class BratIO(IO):
    """
    An IO interface of brat-format files. 
    
    Note: Only support character-based format for Chinese text. 
    """
    def __init__(self, 
                 attr_names=None, 
                 pre_inserted_spaces=True, 
                 tokenize_callback=None, 
                 max_len=500, 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.attr_names = [] if attr_names is None else attr_names
        self.pre_inserted_spaces = pre_inserted_spaces
        self.tokenize_callback = tokenize_callback
        self.max_len = max_len
        self.kwargs = kwargs
        self.line_sep = "\r\n"
        self.sentence_seps = ["。"]
        self.phrase_seps = ["；", "，", ";", ","]
        self.attr_sep = "<a>"
        self.inserted_mark = "😀"
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
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
    
    
    def _segment_text(self, text: str):
        for start, end in segment_text_with_hierarchical_seps(text, 
                                                              hie_seps=[[self.line_sep], self.sentence_seps, self.phrase_seps], 
                                                              length=self.max_len):
            if end - start <= self.max_len:
                yield (start, end)
            else:
                for sub_start, sub_end in segment_text_uniformly(text[start:end], max_span_size=self.max_len):
                    yield (start+sub_start, start+sub_end)
                    
                    
    def read(self, file_path):
        with open(file_path, 'r', encoding=self.encoding) as f:
            text = f.read()
        with open(file_path.replace('.txt', '.ann'), 'r', encoding=self.encoding) as f:
            anns = [ann for ann in f.readlines() if not ann.strip() == ""]
            assert all(ann.startswith(('T', 'A', 'R')) for ann in anns)
            
        if ("\n" in text) and (self.line_sep not in text):
            text = text.replace("\n", self.line_sep)
            
        # Parse chunks
        text_chunks = dict([self._parse_chunk_ann(ann) for ann in anns if ann.startswith('T')])
        if len(text_chunks) == 0:
            return []
        
        text, text_chunks = self._clean_text_chunks(text, text_chunks)
        self._check_text_chunks(text, text_chunks)
        if self.pre_inserted_spaces:
            text, text_chunks = self._remove_pre_inserted_spaces(text, text_chunks)    
            self._check_text_chunks(text, text_chunks)
        
        # Build dataframe
        df = pandas.DataFrame(text_chunks, index=['text', 'type', 'start_in_text', 'end_in_text']).T
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
        for start, end in self._segment_text(text):
            if text[start:end].strip():
                data.append(self._build_entry(text, df, start, end))
        return data
    
    
    def _build_entry(self, text: str, df: pandas.DataFrame, line_start=None, line_end=None):
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
        
        Any single space would be regarded as a inserted space; 
        Any `N` consecutive spaces would be regared as one space (in the original text) with `N-1` inserted spaces. 
        """
        text = re.sub("(?<! ) (?! )", self.inserted_mark, text)
        text = re.sub(" {2,}", lambda x: " " + self.inserted_mark*(len(x.group())-1), text)
        
        is_inserted = [int(c == self.inserted_mark) for c in text]
        num_inserted = numpy.cumsum(is_inserted).tolist()
        # The positions of exact pre-inserted spaces should be mapped to positions NEXT to them
        num_inserted = [n - i for n, i in zip(num_inserted, is_inserted)]
        num_inserted.append(num_inserted[-1])
        
        text = text.replace(self.inserted_mark, "")
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            chunk_start_in_text -= num_inserted[chunk_start_in_text]
            chunk_end_in_text -= num_inserted[chunk_end_in_text]
            assert text[chunk_start_in_text:chunk_end_in_text].replace(" ", "") == chunk_text.replace(" ", "")
            chunk_text = text[chunk_start_in_text:chunk_end_in_text]
            
            text_chunks[chunk_id] = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            
        return text, text_chunks
        
    
    def _check_text_chunks(self, text: str, text_chunks: dict):
        # Check chunk text consistenty
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            assert text[chunk_start_in_text:chunk_end_in_text] == chunk_text
            
    def _clean_text_chunks(self, text: str, text_chunks: dict):
        # Replace `\t` with ` `
        text = re.sub("[ \t]", " ", text)
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            chunk_text = re.sub("[ \t]", " ", chunk_text)
            text_chunks[chunk_id] = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
        return text, text_chunks
        
    
    def write(self, data, file_path):
        text_lines = []
        text_chunks = {}
        
        chunk_idx, attr_idx = 1, 1
        chunk_lines, attr_lines = [], []
        
        line_start = 0
        for data_entry in data:
            line_text = "".join(data_entry['tokens'].raw_text)
            text_lines.append(line_text)
            
            for chunk_type, chunk_start, chunk_end in data_entry['chunks']:
                chunk_text = line_text[chunk_start:chunk_end]
                chunk_start_in_text = chunk_start + line_start
                chunk_end_in_text = chunk_end + line_start
                
                text_chunk = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
                text_chunks[f"T{chunk_idx}"] = text_chunk
                chunk_idx += 1
                
            line_start = line_start + len(line_text) + len(self.line_sep)
            
        text = self.line_sep.join(text_lines)
        
        text, text_chunks = self._clean_text_chunks(text, text_chunks)
        self._check_text_chunks(text, text_chunks)
        if self.pre_inserted_spaces:
            text, text_chunks = self._insert_spaces(text, text_chunks)
            self._check_text_chunks(text, text_chunks)
            
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            
            chunk_type, *attr_values = chunk_type.split(self.attr_sep)
            text_chunk = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            chunk_lines.append(self._build_chunk_ann(chunk_id, text_chunk))
            
            for attr_name, attr_v in zip(self.attr_names, attr_values):
                if attr_v == 'T':
                    attr_lines.append(self._build_attr_ann(f"A{attr_idx}", (attr_name, chunk_id)))
                    attr_idx += 1
                    
        with open(file_path, 'w', encoding=self.encoding) as f:
            f.write(text.replace(self.line_sep, "\n"))
            f.write("\n")
        with open(file_path.replace('.txt', '.ann'), 'w', encoding=self.encoding) as f:
            f.write("\n".join(chunk_lines))
            f.write("\n")
            f.write("\n".join(attr_lines))
            f.write("\n")
            
            
    def _tokenize_and_rejoin(self, text: str):
        tokenized = []
        for tok in self.tokenize_callback(text):
            if (not re.fullmatch("\s+", tok)) and (len(tokenized) > 0):
                tokenized.append(self.inserted_mark)
            tokenized.append(tok)
            
        text = "".join(tokenized)
        text = re.sub(f"[“/]{self.inserted_mark}(?!\s)",    lambda x: x.group().replace(self.inserted_mark, ""), text)
        text = re.sub(f"(?<!\s){self.inserted_mark}[”：/]", lambda x: x.group().replace(self.inserted_mark, ""), text)
        return text
    
    
    def _insert_spaces(self, text: str, text_chunks: dict):
        """
        Chinese text may be tokenized and re-joined by spaces before annotation. 
        
        Any `N` consecutive spaces (in the original text) would be inserted one 
        additional space, resulting in `N+1` consecutive spaces. 
        """
        ori_num_chars = len(text)
        text = self.line_sep.join([self._tokenize_and_rejoin(line) for line in text.split(self.line_sep)])
        
        is_inserted = [int(c == self.inserted_mark) for c in text]
        num_inserted = numpy.cumsum(is_inserted).tolist()
        num_inserted = [n for n, i in zip(num_inserted, is_inserted) if i == 0]
        assert len(num_inserted) == ori_num_chars
        num_inserted.append(num_inserted[-1])
        
        text = text.replace(self.inserted_mark, " ")
        for chunk_id, text_chunk in text_chunks.items():
            chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text = text_chunk
            chunk_start_in_text += num_inserted[chunk_start_in_text]
            chunk_end_in_text += num_inserted[chunk_end_in_text] - int(num_inserted[chunk_end_in_text] > num_inserted[chunk_end_in_text-1])
            assert text[chunk_start_in_text:chunk_end_in_text].replace(" ", "") == chunk_text.replace(" ", "")
            chunk_text = text[chunk_start_in_text:chunk_end_in_text]
            
            text_chunks[chunk_id] = (chunk_text, chunk_type, chunk_start_in_text, chunk_end_in_text)
            
        return text, text_chunks
        
        