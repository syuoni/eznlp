# -*- coding: utf-8 -*-
from collections import Counter
import tqdm
import bs4

from eznlp.token import TokenSequence
from eznlp.io import JsonIO


def parse_tag(tag: bs4.element.Tag):
    tokenized, chunks = [], []
    for subtag in tag.contents:
        if isinstance(subtag, bs4.element.Tag):
            assert subtag.name == 'cons'
            start = len(tokenized)
            sub_tokenized, sub_chunks = parse_tag(subtag)
            
            if 'sem' in subtag.attrs:
                curr_chunk = (subtag['sem'], start, start + len(sub_tokenized), sub_tokenized)
                chunks.append(curr_chunk)
            # else:
            #     print(subtag)
            
            sub_chunks = [(sck[0], start+sck[1], start+sck[2], sck[3]) for sck in sub_chunks]
            chunks.extend(sub_chunks)
            tokenized.extend(sub_tokenized)
        else:
            assert isinstance(subtag, bs4.element.NavigableString)
            tokenized.extend(str(subtag).strip().split(" "))
    return tokenized, chunks


with open("GENIA_term_3.02/GENIAcorpus3.02.xml") as f:
    corpus = [line.strip() for line in f if line.startswith('<sentence>')]

assert all(sent.endswith('</sentence>') for sent in corpus)
# corpus = [sent[len('<sentence>'):-len('</sentence>')] for sent in corpus]


data = []
for sent in tqdm.tqdm(corpus):
    soup = bs4.BeautifulSoup(sent, 'lxml')
    tokenized, chunks = parse_tag(soup.html.body.sentence)
    assert all(ck[3] == tokenized[ck[1]:ck[2]] for ck in chunks)
    
    data.append({'tokens': TokenSequence.from_tokenized_text(tokenized), 
                 'chunks': [(ck[0], ck[1], ck[2]) for ck in chunks]})


def normalize_type(ent_type: str):
    for key in ['G#DNA', 'G#RNA', 'G#protein', 'G#cell_line', 'G#cell_type']:
        if ent_type.startswith(key):
            return key
    else:
        return None


for entry in data:
    entry['chunks'] = [(normalize_type(ck[0]), ck[1], ck[2]) for ck in entry['chunks'] if normalize_type(ck[0]) is not None]

num_train = int(len(data) * 0.9 + 0.5)
train_data = data[:num_train]
test_data  = data[num_train:]
assert len(train_data) + len(test_data) == len(data)

# Counter([ck[0] for ex in test_data for ck in ex['chunks']]).most_common(10)
# sum(len(ex['chunks']) for ex in test_data)

json_io = JsonIO()
json_io.write(train_data, "term.train.json")
json_io.write(test_data, "term.test.json")
