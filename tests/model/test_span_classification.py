# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, SpanClassificationDecoderConfig, ExtractorConfig
from eznlp.training import Trainer


class TestModel(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch = [self.dataset[i] for i in range(4)]
        batch012 = self.dataset.collate(batch[:3]).to(self.device)
        batch123 = self.dataset.collate(batch[1:]).to(self.device)
        losses012, states012 = self.model(batch012, return_states=True)
        losses123, states123 = self.model(batch123, return_states=True)
        
        hidden012, hidden123 = states012['full_hidden'], states123['full_hidden']
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        assert delta_hidden.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        pred012 = self.model.decode(batch012, **states012)
        pred123 = self.model.decode(batch123, **states123)
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
        
        
    @pytest.mark.parametrize("agg_mode", ['max_pooling', 'multiplicative_attention'])
    @pytest.mark.parametrize("size_emb_dim", [25, 0])
    @pytest.mark.parametrize("fl_gamma", [0.0, 2.0])
    def test_model(self, agg_mode, size_emb_dim, fl_gamma, conll2004_demo, device):
        self.config = ExtractorConfig(decoder=SpanClassificationDecoderConfig(agg_mode=agg_mode, size_emb_dim=size_emb_dim, fl_gamma=fl_gamma))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ExtractorConfig('span_classification', ohots=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                      intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, device):
        self.config = ExtractorConfig('span_classification')
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)



@pytest.mark.parametrize("num_neg_chunks", [10, 100])
@pytest.mark.parametrize("max_span_size", [2, 5, 50])
@pytest.mark.parametrize("training", [True, False])
def test_spans_obj(EAR_data_demo, num_neg_chunks, max_span_size, training):
    entry = EAR_data_demo[0]
    tokens, chunks = entry['tokens'], entry['chunks']
    
    config = ExtractorConfig(decoder=SpanClassificationDecoderConfig(num_neg_chunks=num_neg_chunks, max_span_size=max_span_size))
    dataset = Dataset(EAR_data_demo, config, training=training)
    dataset.build_vocabs_and_dims()
    
    spans_obj = dataset[0]['spans_obj']
    assert spans_obj.chunks == chunks
    
    if len(tokens) > max_span_size:
        num_candidate_spans = (len(tokens) - max_span_size)*max_span_size + (max_span_size+1)*max_span_size/2
    else:
        num_candidate_spans = (len(tokens) + 1) * len(tokens) / 2
    
    chunk_spans = [(s, e) for l, s, e in chunks]
    oov_spans = [(s, e) for l, s, e in chunks if e-s > max_span_size]
    if training:
        assert set(chunk_spans).issubset(set(spans_obj.spans))
        assert len(spans_obj.spans) == min(num_candidate_spans + len(oov_spans), len(chunks) + num_neg_chunks)
    else:
        assert set(chunk_spans) - set(spans_obj.spans) == set(oov_spans)
        assert len(spans_obj.spans) == num_candidate_spans
    
    assert (spans_obj.span_size_ids+1).tolist() == [min(e-s, max_span_size) for s, e in spans_obj.spans]
    
    span2label = {(s, e): l for l, s, e in chunks}
    assert all(span2label.get(span, config.decoder.none_label) == config.decoder.idx2label[label_id] 
                   for span, label_id 
                   in zip(spans_obj.spans, spans_obj.label_ids.tolist()))
