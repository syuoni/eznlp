# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.token import TokenSequence
from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, SpanClassificationDecoderConfig, ModelConfig
from eznlp.training import Trainer


class TestModel(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch = [self.dataset[i] for i in range(4)]
        batch012 = self.dataset.collate(batch[:3]).to(self.device)
        batch123 = self.dataset.collate(batch[1:]).to(self.device)
        losses012, hidden012 = self.model(batch012, return_hidden=True)
        losses123, hidden123 = self.model(batch123, return_hidden=True)
        
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        assert delta_hidden.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        pred012 = self.model.decode(batch012)
        pred123 = self.model.decode(batch123)
        assert pred012[1:] == pred123[:-1]
        
        
    def _assert_trainable(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        trainer = Trainer(self.model, optimizer=optimizer, device=self.device)
        dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                 batch_size=4, 
                                                 shuffle=True, 
                                                 collate_fn=self.dataset.collate)
        trainer.train_epoch(dataloader)
        
        
    def _setup_case(self, data, device):
        self.device = device
        
        self.dataset = Dataset(data, self.config)
        self.dataset.build_vocabs_and_dims()
        self.model = self.config.instantiate().to(self.device)
        assert isinstance(self.config.name, str) and len(self.config.name) > 0
        
        
    @pytest.mark.parametrize("enc_arch", ['Conv', 'LSTM'])
    @pytest.mark.parametrize("agg_mode", ['max_pooling'])
    def test_model(self, enc_arch, agg_mode, conll2004_demo, device):
        self.config = ModelConfig(intermediate2=EncoderConfig(arch=enc_arch), 
                                  decoder=SpanClassificationDecoderConfig(agg_mode=agg_mode))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('span_classification', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()




@pytest.mark.parametrize("num_neg_chunks, max_span_size, training", [(10, 5, True), 
                                                                     (10, 5, False), 
                                                                     (100, 20, True), 
                                                                     (100, 20, False)])
def test_spans_obj(num_neg_chunks, max_span_size, training):
    tokenized_raw_text = ["This", "is", "a", "-3.14", "demo", ".", 
                          "Those", "are", "an", "APPLE", "and", "some", "glass", "bottles", "."]
    tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
    
    entities = [{'type': 'EntA', 'start': 4, 'end': 5},
                {'type': 'EntA', 'start': 9, 'end': 10},
                {'type': 'EntB', 'start': 12, 'end': 14}]
    chunks = [(ent['type'], ent['start'], ent['end']) for ent in entities]
    data = [{'tokens': tokens, 'chunks': chunks}]
    
    config = ModelConfig(decoder=SpanClassificationDecoderConfig(num_neg_chunks=num_neg_chunks, 
                                                                 max_span_size=max_span_size))
    dataset = Dataset(data, config, training=training)
    dataset.build_vocabs_and_dims()
    
    spans_obj = dataset[0]['spans_obj']
    assert spans_obj.chunks == chunks
    assert set((ck[1], ck[2]) for ck in chunks).issubset(set(spans_obj.spans))
    assert (spans_obj.span_size_ids+1).tolist() == [e-s for s, e in spans_obj.spans]
    
    num_tokens = len(tokens)
    if num_tokens > max_span_size:
        expected_num_spans = (num_tokens-max_span_size)*max_span_size + (max_span_size+1)*max_span_size/2
    else:
        expected_num_spans = (num_tokens+1)*num_tokens / 2
    if training:
        expected_num_spans = min(expected_num_spans, len(chunks) + num_neg_chunks)
    assert len(spans_obj.spans) == expected_num_spans
    
    if training:
        assert (spans_obj.label_ids[:len(chunks)] != config.decoder.none_idx).all().item()
        assert (spans_obj.label_ids[len(chunks):] == config.decoder.none_idx).all().item()
    else:
        assert not hasattr(spans_obj, 'label_ids')
