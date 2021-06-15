# -*- coding: utf-8 -*-
from eznlp.token import TokenSequence
from eznlp.utils import ChunksTagsTranslator, TextChunksTranslator


def test_tags2text_chunks(spacy_nlp_en):
    raw_text = "This is a -3.14 demo. Those are an APPLE and some glass bottles."
    tokens = TokenSequence.from_raw_text(raw_text, spacy_nlp_en)
    
    entities = [{'entity': 'demo', 'type': 'EntA', 'start': 16, 'end': 20},
                {'entity': 'APPLE', 'type': 'EntA', 'start': 35, 'end': 40},
                {'entity': 'glass bottles', 'type': 'EntB', 'start': 50, 'end': 63}]
    text_chunks = [(ent['type'], ent['start'], ent['end'], ent['entity']) for ent in entities]
    tags = ['O', 'O', 'O', 'O', 'S-EntA', 'O', 'O', 'O', 'O', 'S-EntA', 'O', 'O', 'B-EntB', 'E-EntC', 'O']
    
    # `breaking_for_types` is False
    text_translator = TextChunksTranslator()
    tags_translator = ChunksTagsTranslator(scheme='BIOES', breaking_for_types=False)
    chunks, *_ = text_translator.text_chunks2chunks(text_chunks, tokens, raw_text)
    tags_built = tags_translator.chunks2tags(chunks, len(tokens))
    assert tags_built[-2] == 'E-EntB'
    tags_built[-2] = 'E-EntC'
    assert tags_built == tags
    
    chunks = tags_translator.tags2chunks(tags)
    text_chunks_retr = text_translator.chunks2text_chunks(chunks, tokens, raw_text, append_chunk_text=True)
    assert text_chunks_retr == text_chunks
    
    
    # Spans?
    text_chunks_retr_spans = []
    for span in tokens.spans_within_max_length(10):
        curr_chunks = tags_translator.tags2chunks(tags[span])
        curr_text_chunks = text_translator.chunks2text_chunks(curr_chunks, tokens[span], raw_text, append_chunk_text=True)
        text_chunks_retr_spans.extend(curr_text_chunks)
    assert text_chunks_retr_spans == text_chunks
    
    
    # `breaking_for_types` is True
    tags_translator = ChunksTagsTranslator(scheme='BIOES', breaking_for_types=True)
    chunks = tags_translator.tags2chunks(tags)
    text_chunks_retr = text_translator.chunks2text_chunks(chunks, tokens, raw_text, append_chunk_text=True)
    assert text_chunks_retr[:2] == text_chunks[:2]
    assert text_chunks_retr[2][0] == 'EntB'
    assert text_chunks_retr[3][0] == 'EntC'
