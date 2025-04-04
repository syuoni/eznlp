# -*- coding: utf-8 -*-
import re

import bs4
import tqdm

from eznlp.io import JsonIO
from eznlp.token import TokenSequence


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
            if len(text := str(subtag).strip()) > 0:
                tokenized.extend(re.split(" +", text))
    return tokenized, chunks


with open("GENIA_term_3.02/GENIAcorpus3.02.xml") as f:
    corpus = f.read()

corpus = corpus.split("\n\n")


data = []
doc_key = 0
for doc in tqdm.tqdm(corpus):
    if doc.startswith("<article>"):
        assert doc.endswith("</article>")
        soup = bs4.BeautifulSoup(doc, 'lxml')
        bibliomisc = soup.find("bibliomisc").text
        for sent in soup.find_all('sentence'):
            tokenized, chunks = parse_tag(sent)
            assert "" not in tokenized
            assert all(ck[3] == tokenized[ck[1]:ck[2]] for ck in chunks)

            data.append({'bibliomisc': bibliomisc,
                         'doc_key': f"genia_3.02_article_{doc_key}",
                         'tokens': TokenSequence.from_tokenized_text(tokenized),
                         'chunks': [(ck[0], ck[1], ck[2]) for ck in chunks]})

        doc_key += 1

    else:
        assert not doc.endswith("</article>")



def normalize_type(ent_type: str):
    for key in ['G#DNA', 'G#RNA', 'G#protein', 'G#cell_line', 'G#cell_type']:
        if ent_type.startswith(key):
            return key
    else:
        return None


for entry in data:
    entry['chunks'] = [(normalize_type(ck[0]), ck[1], ck[2]) for ck in entry['chunks'] if normalize_type(ck[0]) is not None]



num_train, num_dev, num_test = 15_023, 1_669, 1_854
train_data = data[:num_train]
dev_data   = data[num_train:num_train+num_dev]
test_data  = data[num_train+num_dev:]
assert len(train_data) + len(dev_data) + len(test_data) == len(data)
assert len(test_data) == num_test

# Counter([ck[0] for ex in test_data for ck in ex['chunks']]).most_common(10)
# sum(len(ex['chunks']) for ex in test_data)

json_io = JsonIO(retain_meta=True)
json_io.write(train_data, "term.train.json")
json_io.write(dev_data,   "term.dev.json")
json_io.write(test_data,  "term.test.json")
