# -*- coding: utf-8 -*-
import json


task = 'ace05'
cv = 4
partition = 'test'


for partition in ['train', 'dev', 'test']:
    if task == 'ace05':
        src_fn = f"{task}/json/{partition}.json"
        trg_fn = f"{task}/{partition}.json"
    else:
        if partition == 'dev':
            continue
        src_fn = f"{task}/json/{partition}/{cv}.json"
        trg_fn = f"{task}/cv{cv}.{partition}.json"
    
    with open(src_fn) as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    new_data = []
    for ex in data:
        curr_start = 0
        
        for k in range(len(ex['sentences'])):
            new_ex = {}
            new_ex['doc_key'] = ex['doc_key']
            new_ex['tokens'] = ex['sentences'][k]
            new_ex['entities'] = [{'start': ent[0]-curr_start, 'end': ent[1]-curr_start+1, 'type': ent[2]} for ent in ex['ner'][k]]
            
            spans = [(ent['start'], ent['end']) for ent in new_ex['entities']]
            new_ex['relations'] = [{'type': rel[4], 
                                    'head': spans.index((rel[0]-curr_start, rel[1]-curr_start+1)), 
                                    'tail': spans.index((rel[2]-curr_start, rel[3]-curr_start+1))} for rel in ex['relations'][k]]
            new_data.append(new_ex)
            curr_start += len(new_ex['tokens'])
    
    
    with open(trg_fn, 'w') as f:
        json.dump(new_data, f, ensure_ascii=False)
