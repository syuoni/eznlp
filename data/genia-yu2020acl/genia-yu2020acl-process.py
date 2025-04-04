# -*- coding: utf-8 -*-
import json

partition = "test"

with open(f"{partition}.genia.jsonlines") as f:
    data = [json.loads(line) for line in f]


new_data = []
for entry in data:
    assert len(entry["sentences"]) == len(entry["ners"])
    for sent, ner in zip(entry["sentences"], entry["ners"]):
        new_entry = {
            "doc_key": entry["doc_key"],
            "tokens": sent,
            "entities": [
                {"start": ent[0], "end": ent[1] + 1, "type": ent[2]} for ent in ner
            ],
        }
        new_data.append(new_entry)


with open(f"{partition}.json", "w") as f:
    json.dump(new_data, f, ensure_ascii=False)
