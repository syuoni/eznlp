# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import BertLikeConfig, SpanBertLikeConfig, SpecificSpanClsDecoderConfig, SpecificSpanExtractorConfig
from eznlp.model.decoder.boundary_selection import _spans_from_upper_triangular
from eznlp.model.decoder.specific_span_classification import _spans_from_diagonals
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
        
        # Check the consistency of query hidden states
        for k in range(2, self.config.decoder.max_span_size+1):
            query012, query123 = states012['all_query_hidden'][k], states123['all_query_hidden'][k]
            delta_query = query012[1:, :min_step-k+1] - query123[:-1, :min_step-k+1]
            assert delta_query.abs().max().item() < 1e-4
        
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
        
        
    @pytest.mark.parametrize("num_layers, share_weights, agg_mode, size_emb_dim", 
                             [(12, False, 'max_pooling', 0), 
                              (6,  False, 'max_pooling', 0), 
                              (1,  False, 'max_pooling', 0), 
                              (12, True,  'max_pooling', 0), 
                              (12, False, 'mean_pooling', 0), 
                              (12, False, 'multiplicative_attention', 0), 
                              (12, False, 'max_pooling', 25)])
    def test_model(self, num_layers, share_weights, agg_mode, size_emb_dim, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(decoder=SpecificSpanClsDecoderConfig(size_emb_dim=size_emb_dim), 
                                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, output_hidden_states=True), 
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert, freeze=False, num_layers=num_layers, share_weights=share_weights, init_agg_mode=agg_mode), 
                                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        if share_weights:
            assert len(self.model.span_bert_like.query_bert_like.layer) == num_layers
        else:
            assert len(self.model.span_bert_like.query_bert_like[0].layer) == num_layers
        
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, output_hidden_states=True), 
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert), 
                                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)



@pytest.mark.parametrize("seq_len, max_span_size", [(5, 1),   (5, 5), 
                                                    (10, 1),  (10, 5), (10, 10), 
                                                    (100, 1), (100, 10), (100, 100)])
def test_spans_from_diagonals(seq_len, max_span_size):
    assert len(list(_spans_from_diagonals(seq_len, max_span_size))) == (seq_len*2-max_span_size+1)*max_span_size // 2


@pytest.mark.parametrize("seq_len", [5, 10, 100])
def test_spans_from_functions(seq_len):
    assert set(_spans_from_diagonals(seq_len)) == set(_spans_from_upper_triangular(seq_len))
