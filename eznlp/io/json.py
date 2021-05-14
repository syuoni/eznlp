# -*- coding: utf-8 -*-
from typing import List
import logging
import json

from ..token import TokenSequence
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
    """
    An IO Interface of Json files. 
    
    """
    def __init__(self, 
                 text_key='tokens', 
                 chunk_key='entities', 
                 chunk_type_key='type', 
                 chunk_start_key='start', 
                 chunk_end_key='end', 
                 relation_key='relations', 
                 relation_type_key='type', 
                 relation_head_key='head', 
                 relation_tail_key='tail', 
                 drop_duplicated=True, 
                 encoding=None, 
                 verbose: bool=True, 
                 **kwargs):
        self.text_key = text_key
        self.chunk_key = chunk_key
        self.chunk_type_key = chunk_type_key
        self.chunk_start_key = chunk_start_key
        self.chunk_end_key = chunk_end_key
        if all(key is not None for key in [relation_key, relation_type_key, relation_head_key, relation_tail_key]):
            self.relation_key = relation_key
            self.relation_type_key = relation_type_key
            self.relation_head_key = relation_head_key
            self.relation_tail_key = relation_tail_key
        else:
            self.relation_key = None
        self.drop_duplicated = drop_duplicated
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
        
    def read(self, file_path):
        with open(file_path, 'r', encoding=self.encoding) as f:
            raw_data = json.load(f)
        
        data = []
        for raw_entry in raw_data:
            tokens = TokenSequence.from_tokenized_text(raw_entry[self.text_key], **self.kwargs)
            chunks = [(chunk[self.chunk_type_key], 
                       chunk[self.chunk_start_key],
                       chunk[self.chunk_end_key]) for chunk in raw_entry[self.chunk_key]]
            chunks = _filter_duplicated(chunks) if self.drop_duplicated else chunks
            data_entry = {'tokens': tokens, 'chunks': chunks}
            
            if self.relation_key is not None:
                relations = [(rel[self.relation_type_key], 
                              chunks[rel[self.relation_head_key]], 
                              chunks[rel[self.relation_tail_key]]) for rel in raw_entry[self.relation_key]]
                relations = _filter_duplicated(relations) if self.drop_duplicated else relations
                data_entry.update({'relations': relations})
                
            data.append(data_entry)
            
        return data




class SQuADIO(IO):
    def __init__(self, tokenize_callback=None, encoding=None, verbose: bool=True, **kwargs):
        self.translator = TextChunksTranslator(has_chunk_text=True)
        self.tokenize_callback = tokenize_callback
        super().__init__(encoding=encoding, verbose=verbose, **kwargs)
        
    def read(self, file_path, return_errors: bool=False):
        with open(file_path, 'r', encoding=self.encoding) as f:
            raw_data = json.load(f)
        raw_data = raw_data['data']
        
        data = []
        errors, mismatches = [], []
        for doc in raw_data:
            for parag in doc['paragraphs']:
                context = TokenSequence.from_raw_text(parag['context'], self.tokenize_callback, **self.kwargs)
                
                for qas in parag['qas']:
                    assert qas['is_impossible'] or len(qas['answers']) > 0
                    
                    question = TokenSequence.from_raw_text(qas['question'], self.tokenize_callback, **self.kwargs)
                    answers = [('<ans>', ans['answer_start'], ans['answer_start']+len(ans['text']), ans['text']) for ans in qas['answers']]
                    answers, curr_errors, curr_mismatches = self.translator.text_chunks2chunks(answers, context, parag['context'])
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
