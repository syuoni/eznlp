# -*- coding: utf-8 -*-
from ..token import TokenSequence
from .transition import SchemeTranslator


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

