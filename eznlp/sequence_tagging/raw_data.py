# -*- coding: utf-8 -*-
from eznlp import TokenSequence

def parse_conll_file(file_path, 
                     columns=['text', 'pos_tag', 'chunking_tag'], text_col='text', trg_col='chunking_tag', 
                     attach_additional_tags=False):
    data = []
    example = [[] for c in columns]
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            
            if line == '':
                if len(example[0]) == 0:
                    continue
                
                tokenized_text = example[columns.index(text_col)]
                tags = example[columns.index(trg_col)]
                additional_tags = {c: ex_part for c, ex_part in zip(columns, example) if c not in (text_col, trg_col)}
                
                tokens = TokenSequence.from_tokenized_text(tokenized_text, additional_tags=additional_tags)
                data.append({'tokens': tokens, 
                             'tags': tags})
                example = [[] for c in columns]
            else:
                for ex_part, ex_part_to_append in zip(example, line.split(' ')):
                    ex_part.append(ex_part_to_append)
    return data
