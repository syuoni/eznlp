# -*- coding: utf-8 -*-
import re
import os
import glob
import eznlp

translator = eznlp.utils.ChunksTagsTranslator(scheme='OntoNotes')

def parse_sentence(sent):
    sent = sent.strip()
    tokenized, entities = [], []
    start = 0
    for ent in re.finditer("<ENAMEX TYPE=\"\w+\".*?>.+?</ENAMEX>", sent):
        if start < ent.start() and len(sent[start:ent.start()].strip()) > 0:
            tokenized.extend(sent[start:ent.start()].strip().split(" "))

        ent_type = re.search("(?<=<ENAMEX TYPE=\")\w+(?=\".*?>)", ent.group()).group()
        ent_text = re.search("(?<=>).+(?=</ENAMEX>)", ent.group()).group()
        ent_text_tokenized = ent_text.split(" ")

        entities.append((ent_type, len(tokenized), len(tokenized) + len(ent_text_tokenized)))
        tokenized.extend(ent_text_tokenized)
        start = ent.end()

    if start < len(sent) and len(sent[start:].strip()) > 0:
        tokenized.extend(sent[start:].strip().split(" "))

    tags = translator.chunks2tags(entities, len(tokenized))
    entities_retr = translator.tags2chunks(tags)
    assert entities_retr == entities
    assert len(tags) == len(tokenized)
    return tokenized, tags


files = glob.glob("../ontonotes-release-4.0/data/files/data/chinese/annotations/*/*/*/*.name")
print(len(files))

# Che et al. (2013)
# This corpus includes about 400 document pairs (chtb 0001-0325, ectb 1001-1078).
# We used odd numbered documents as development data and even numbered documents as test data. 
# We used all other portions of the named entity annotated corpus as training data.
dev_files = [fn for fn in files if os.path.basename(fn).startswith('chtb') and int(re.search('\d+', os.path.basename(fn)).group()) % 2 == 1]
test_files = [fn for fn in files if os.path.basename(fn).startswith('chtb') and int(re.search('\d+', os.path.basename(fn)).group()) % 2 == 0]
train_files = [fn for fn in files if fn not in dev_files + test_files]


for split_name, split_files in zip(['dev', 'test', 'train'], [dev_files, test_files, train_files]):
    with open(f"{split_name}.chinese.vz_gold_conll", 'w', encoding='utf-8') as wf:
        for fn in split_files:
            with open(fn, encoding='utf-8') as f:
                doc = f.readlines()
            assert doc[0].startswith("<DOC")
            assert doc[-1].startswith("</DOC")
            
            doc_name = '/'.join(files[0].split('\\')[1:]).replace('.name', '')
            wf.write("#begin document ({})\n".format(doc_name))
            for sent in doc[1:-1]:
                tokenized, tags = parse_sentence(sent)
                for k, (token, tag) in enumerate(zip(tokenized, tags)):
                    wf.write("{}\t{}\t{}\t{}\n".format(doc_name, k, token, tag))
                wf.write("\n")
            wf.write("#end document\n")
