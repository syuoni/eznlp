# -*- coding: utf-8 -*-
import pytest

from eznlp.token import TokenSequence
from eznlp.span_classification import SpanClassificationDecoderConfig, SpanClassifierConfig
from eznlp.span_classification import SpanClassificationDataset


class TestSpanClassificationDataset(object):
    @pytest.mark.parametrize("num_neg_chunks, max_span_size, training", [(10, 5, True), 
                                                                         (10, 5, False), 
                                                                         (100, 20, True), 
                                                                         (100, 20, False)])
    def test_spans(self, num_neg_chunks, max_span_size, training):
        tokenized_raw_text = ["This", "is", "a", "-3.14", "demo", ".", 
                              "Those", "are", "an", "APPLE", "and", "some", "glass", "bottles", "."]
        tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
        
        entities = [{'type': 'EntA', 'start': 4, 'end': 5},
                    {'type': 'EntA', 'start': 9, 'end': 10},
                    {'type': 'EntB', 'start': 12, 'end': 14}]
        chunks = [(ent['type'], ent['start'], ent['end']) for ent in entities]
        data = [{'tokens': tokens, 'chunks': chunks}]
        
        config = SpanClassifierConfig(decoder=SpanClassificationDecoderConfig(num_neg_chunks=num_neg_chunks, 
                                                                              max_span_size=max_span_size))
        dataset = SpanClassificationDataset(data, config, training=training)
        dataset.build_vocabs_and_dims()
        
        spans_obj = dataset[0]['spans_obj']
        assert spans_obj.chunks == chunks
        assert set((ck[1], ck[2]) for ck in chunks).issubset(set(spans_obj.spans))
        assert (spans_obj.span_size_ids+1).tolist() == [e-s for s, e in spans_obj.spans]
        assert (spans_obj.label_ids[:len(chunks)] != config.decoder.none_idx).all().item()
        assert (spans_obj.label_ids[len(chunks):] == config.decoder.none_idx).all().item()
        
        num_tokens = len(tokens)
        if num_tokens > max_span_size:
            expected_num_spans = (num_tokens-max_span_size)*max_span_size + (max_span_size+1)*max_span_size/2
        else:
            expected_num_spans = (num_tokens+1)*num_tokens / 2
        if training:
            expected_num_spans = min(expected_num_spans, len(chunks) + num_neg_chunks)
            
        assert len(spans_obj.spans) == expected_num_spans
        