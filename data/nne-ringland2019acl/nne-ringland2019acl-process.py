# -*- coding: utf-8 -*-
import json


for partition in ['train', 'dev', 'test']:
    with open(f"{partition}.txt") as f:
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
                start, end = [int(x) for x in positions.split(',')]
                entry['entities'].append({'type': ent_type, 'start': start, 'end': end+1})

    data = [{'tokens': entry['tokens'], 'entities': entry['entities']} for entry in data]

    with open(f"{partition}.json", 'w') as f:
        json.dump(data, f, ensure_ascii=False)
