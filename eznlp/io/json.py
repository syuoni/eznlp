# -*- coding: utf-8 -*-
from typing import List
import logging
import json

from ..utils import TextChunksTranslator
from .base import IO

logger = logging.getLogger(__name__)


def _filter_duplicated(tuples: List[tuple]):
    filtered_tuples = []
    for tp in tuples:
        if tp not in filtered_tuples:
            filtered_tuples.append(tp)
    return filtered_tuples


class JsonIO(IO):
    """An IO Interface of Json files. 
    
    """
    def __init__(self, 
                 is_tokenized: bool=True, 
                 tokenize_callback=None, 
                 text_key='tokens', 
                 chunk_key='entities', 
                 chunk_type_key='type', 
                 chunk_start_key='start', 
                 chunk_end_key='end', 
                 chunk_text_key=None, 
                 relation_key=None, 
                 relation_type_key=None, 
                 relation_head_key=None, 
                 relation_tail_key=None, 
                 drop_duplicated=True, 
                 is_whole_piece: bool=True, 
                 encoding=None, 
                 verbose: bool=True, 
                 **token_kwargs):
        self.text_key = text_key
        self.chunk_key = chunk_key
        self.chunk_type_key = chunk_type_key
        self.chunk_start_key = chunk_start_key
        self.chunk_end_key = chunk_end_key
        self.chunk_text_key = chunk_text_key
        if all(key is not None for key in [relation_key, relation_type_key, relation_head_key, relation_tail_key]):
            self.relation_key = relation_key
            self.relation_type_key = relation_type_key
            self.relation_head_key = relation_head_key
            self.relation_tail_key = relation_tail_key
        else:
            self.relation_key = None
        
        self.drop_duplicated = drop_duplicated
        self.is_whole_piece = is_whole_piece
        
        super().__init__(is_tokenized=is_tokenized, tokenize_callback=tokenize_callback, encoding=encoding, verbose=verbose, **token_kwargs)
        if not self.is_tokenized:
            self.text_translator = TextChunksTranslator()
        
        
    def read(self, file_path, return_errors: bool=False):
        with open(file_path, 'r', encoding=self.encoding) as f:
            if self.is_whole_piece:
                raw_data = json.load(f)
            else:
                raw_data = [json.loads(line) for line in f if len(line.strip()) > 0]
        
        data = []
        errors, mismatches = [], []
        for raw_entry in raw_data:
            tokens = self._build_tokens(raw_entry[self.text_key])
            chunks = [(chunk[self.chunk_type_key], 
                       chunk[self.chunk_start_key],
                       chunk[self.chunk_end_key]) for chunk in raw_entry[self.chunk_key]]
            
            if not self.is_tokenized:
                if self.chunk_text_key is not None:
                    chunks = [(*ck, chunk[self.chunk_text_key]) for ck, chunk in zip(chunks, raw_entry[self.chunk_key])]
                
                chunks, curr_errors, curr_mismatches = self.text_translator.text_chunks2chunks(chunks, tokens, raw_entry[self.text_key], place_none_for_errors=True)
                errors.extend(curr_errors)
                mismatches.extend(curr_mismatches)
            
            if self.drop_duplicated:
                chunks = _filter_duplicated(chunks)
            
            data_entry = {'tokens': tokens, 'chunks': [ck for ck in chunks if ck is not None]}
            
            if self.relation_key is not None:
                relations = [(rel[self.relation_type_key], 
                              chunks[rel[self.relation_head_key]], 
                              chunks[rel[self.relation_tail_key]]) for rel in raw_entry[self.relation_key]]
                relations = _filter_duplicated(relations) if self.drop_duplicated else relations
                data_entry.update({'relations': [(rel_type, head, tail) for rel_type, head, tail in relations if head is not None and tail is not None]})
                
            data.append(data_entry)
            
        if len(errors) > 0 or len(mismatches) > 0:
            logger.info(f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing...")
        
        if return_errors:
            return data, errors, mismatches
        else:
            return data




class SQuADIO(IO):
    """An IO Interface of SQuAD files. 
    
    """
    def __init__(self, tokenize_callback=None, encoding=None, verbose: bool=True, **token_kwargs):
        super().__init__(is_tokenized=False, tokenize_callback=tokenize_callback, encoding=encoding, verbose=verbose, **token_kwargs)
        self.text_translator = TextChunksTranslator()
        
        
    def read(self, file_path, return_errors: bool=False):
        with open(file_path, 'r', encoding=self.encoding) as f:
            raw_data = json.load(f)
        raw_data = raw_data['data']
        
        data = []
        errors, mismatches = [], []
        for doc in raw_data:
            for parag in doc['paragraphs']:
                context = self._build_tokens(parag['context'])
                
                for qas in parag['qas']:
                    assert qas['is_impossible'] or len(qas['answers']) > 0
                    
                    question = self._build_tokens(qas['question'])
                    answers = [('<ans>', ans['answer_start'], ans['answer_start']+len(ans['text']), ans['text']) for ans in qas['answers']]
                    answers, curr_errors, curr_mismatches = self.text_translator.text_chunks2chunks(answers, context, parag['context'])
                    answers = _filter_duplicated(answers)
                    errors.extend(curr_errors)
                    mismatches.extend(curr_mismatches)
                    data.append({'id': qas['id'], 
                                 'title': doc['title'], 
                                 'context': context, 
                                 'question': question, 
                                 'answers': answers})
        
        if len(errors) > 0 or len(mismatches) > 0:
            logger.info(f"{len(errors)} errors and {len(mismatches)} mismatches detected during parsing...")
        
        if return_errors:
            return data, errors, mismatches
        else:
            return data
