# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.model import EncoderConfig
from eznlp.pretrained import BertLikeConfig
from eznlp.text_classification import TextClassificationDecoderConfig, TextClassifierConfig
from eznlp.text_classification import TextClassificationDataset
from eznlp.text_classification import TextClassificationTrainer


class TestClassifier(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch012 = self.dataset.collate([self.dataset[i] for i in range(0, 3)]).to(self.device)
        batch123 = self.dataset.collate([self.dataset[i] for i in range(1, 4)]).to(self.device)
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
        trainer = TextClassificationTrainer(self.model, optimizer=optimizer, device=self.device)
        dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                 batch_size=4, 
                                                 shuffle=True, 
                                                 collate_fn=self.dataset.collate)
        trainer.train_epoch(dataloader)
        
        
    def _setup_case(self, data, device):
        self.device = device
        
        self.dataset = TextClassificationDataset(data, self.config)
        self.dataset.build_vocabs_and_dims()
        self.model = self.config.instantiate().to(self.device)
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    @pytest.mark.parametrize("pooling_mode", ['Min', 'Max', 'Mean'])
    def test_classifier_with_pooling(self, enc_arch, pooling_mode, yelp2013_demo, device):
        self.config = TextClassifierConfig(encoder=EncoderConfig(arch=enc_arch), 
                                           decoder=TextClassificationDecoderConfig(use_attention=False, 
                                                                                   pooling_mode=pooling_mode))
        self._setup_case(yelp2013_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    @pytest.mark.parametrize("attention_scoring", ['Dot', 'Multiplicative', 'Additive'])
    def test_classifier_attention(self, enc_arch, attention_scoring, yelp2013_demo, device):
        self.config = TextClassifierConfig(encoder=EncoderConfig(arch=enc_arch), 
                                           decoder=TextClassificationDecoderConfig(use_attention=True, 
                                                                                   attention_scoring=attention_scoring))
        self._setup_case(yelp2013_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("from_tokenized", [True, False])
    def test_classifier_with_bert_like(self, from_tokenized, yelp2013_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = TextClassifierConfig(ohots=None, 
                                           encoder=None, 
                                           bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert))
        self._setup_case(yelp2013_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        