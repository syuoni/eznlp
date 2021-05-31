# -*- coding: utf-8 -*-
import logging

from ..utils import TextChunksTranslator
from .base import IO

logger = logging.getLogger(__name__)


class ChipIO(IO):
    """An IO interface of CHIP-format files. 

    http://www.cips-chip.org.cn/2020/eval1
    """
    def __init__(self, 
                 tokenize_callback='char', 
                 sep='|||', 
                 encoding=None, 
                 verbose: bool=True, 
                 **token_kwargs):
        self.sep = sep
        super().__init__(is_tokenized=False, tokenize_callback=tokenize_callback, encoding=encoding, verbose=verbose, **token_kwargs)
        self.text_translator = TextChunksTranslator()

        
    def read(self, file_path, return_errors: bool=False):
        with open(file_path, 'r', encoding=self.encoding) as f:
            raw_lines = [line for line in f if len(line.strip()) > 0]

        data = []
        errors, mismatches = [], []
        for line in raw_lines:
            text, *text_chunks = line.strip().split(self.sep)
            tokens = self._build_tokens(text)
            
            text_chunks = [text_chunk.split() for text_chunk in text_chunks if len(text_chunk) > 0]
            text_chunks = [(chunk_type, int(start), int(end)+1) for start, end, chunk_type in text_chunks]

            chunks, curr_errors, curr_mismatches = self.text_translator.text_chunks2chunks(text_chunks, tokens, text, place_none_for_errors=False)
            errors.extend(curr_errors)
            mismatches.extend(curr_mismatches)
            data.append({'tokens': tokens, 'chunks': chunks})

        if len(errors) > 0 or len(mismatches) > 0:
            logger.warning(f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing {file_path}")
        
        if return_errors:
            return data, errors, mismatches
        else:
            return data
