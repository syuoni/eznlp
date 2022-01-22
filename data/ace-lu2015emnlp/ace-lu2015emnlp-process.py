# -*- coding: utf-8 -*-
import json


task = 'ACE2004'
partition = 'test'

for task in ['ACE2004', 'ACE2005']:
    for partition in ['train', 'dev', 'test']:
        with open(f"{task}/{partition}.data") as f:
            lines = [line.strip() for line in f]
        
        data = [{'raw_text': lines[i*4], 
                 'raw_pos': lines[i*4+1], 
                 'raw_entities': lines[i*4+2]} for i in range(len(lines)//4)]
        
        for entry in data:
            entry['tokens'] = entry['raw_text'].split()
            assert len(entry['tokens']) == len(entry['raw_pos'].split())
            
            entry['entities'] = []
            if len(entry['raw_entities']) > 0:
                for entity in entry['raw_entities'].split('|'):
                    positions, ent_type = entity.split()
                    start1, end1, start2, end2 = [int(x) for x in positions.split(',')]
                    entry['entities'].append({'type': ent_type, 'start': start1, 'end': end1})
                    # if (start2, end2) != (start1, end1):
                    #     entry['entities'].append({'type': ent_type, 'start': start2, 'end': end2})
        
        data = [{'tokens': entry['tokens'], 'entities': entry['entities']} for entry in data]
        
        # sum(len(entry['entities']) for entry in data)
        # sum(ent['end']-ent['start'] > 6 for entry in data for ent in entry['entities'])
        # max(ent['end']-ent['start'] for entry in data for ent in entry['entities'])
        
        with open(f"{task}/{partition}.json", 'w') as f:
            json.dump(data, f, ensure_ascii=False)
