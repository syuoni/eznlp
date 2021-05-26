# -*- coding: utf-8 -*-
import os
import shutil
import re


for folder_name in ['ner', 'relation_extraction']:
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    shutil.copytree(f"original/{folder_name}", folder_name)




def _make_repl(offset: int=0, offset_start: int=0):
    def repl(x):
        x_str = x.group()
        x_int = int(x_str)
        if x_int < offset_start:
            return x_str
        else:
            return str(x_int + offset)
    return repl


for file_path, offset, offset_start in [("relation_extraction/Training/1122.txt", -1, 2603), 
                                        ("relation_extraction/Training/191.txt", -1, 706), 
                                        ("relation_extraction/Training/293.txt", -1, 773), 
                                        ("relation_extraction/Training/706.txt", -1, 1005), 
                                        ("relation_extraction/Training/893.txt", -1, 1133), 
                                        ("relation_extraction/Training/898.txt", -1, 329), 
                                        ("relation_extraction/Training/898.txt", -1, 2993), # Two fixes on this file
                                        ("relation_extraction/Training/905.txt", -1, 2511), 
                                        ("relation_extraction/Training/907.txt", -1, 2992)]:
    repl = _make_repl(offset, offset_start)
    
    with open(file_path.replace('.txt', '.ann'), 'r', encoding='utf-8') as f:
        anns = f.readlines()
    
    for k, ann in enumerate(anns):
        if ann.startswith('T'):
            chunk_id, chunk_type_pos, chunk_text = ann.split('\t')
            new_chunk_type_pos = re.sub(pattern='\d+', repl=repl, string=chunk_type_pos)
            anns[k] = '\t'.join([chunk_id, new_chunk_type_pos, chunk_text])
    
    with open(file_path.replace('.txt', '.ann'), 'w', encoding='utf-8') as f:
        f.write("".join(anns))



for file_path, drop_indexes in [("relation_extraction/Training/1108.txt", [292]), 
                                ("relation_extraction/Training/959.txt", list(range(197,225))), 
                                ("relation_extraction/Training/980.txt", [143])]:
    with open(file_path.replace('.txt', '.ann'), 'r', encoding='utf-8') as f:
        anns = f.readlines()
    
    anns = [ann for k, ann in enumerate(anns) if not k in drop_indexes]
    
    with open(file_path.replace('.txt', '.ann'), 'w', encoding='utf-8') as f:
        f.write("".join(anns))

