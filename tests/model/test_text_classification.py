# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, TextClassificationDecoderConfig, ModelConfig
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
        
        
    @pytest.mark.parametrize("arch", ['Conv', 'LSTM'])
    @pytest.mark.parametrize("agg_mode", ['min_pooling', 'max_pooling', 'mean_pooling', 
                                          'dot_attention', 'multiplicative_attention', 'additive_attention'])
    @pytest.mark.parametrize("criterion", ['cross_entropy', 'focal', 'label_smooth'])
    def test_model(self, arch, agg_mode, criterion, yelp_full_demo, device):
        self.config = ModelConfig(intermediate2=EncoderConfig(arch=arch), 
                                  decoder=TextClassificationDecoderConfig(agg_mode=agg_mode, criterion=criterion))
        self._setup_case(yelp_full_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("from_tokenized", [True, False])
    def test_model_with_bert_like(self, from_tokenized, yelp_full_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('text_classification', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, from_tokenized=from_tokenized), 
                                  intermediate2=None)
        self._setup_case(yelp_full_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, yelp_full_demo, device):
        self.config = ModelConfig('text_classification')
        self._setup_case(yelp_full_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in yelp_full_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_labels_pred = trainer.predict(dataset_wo_gold)
        assert len(set_labels_pred) == len(data_wo_gold)
        