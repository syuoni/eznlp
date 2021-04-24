# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.data import Dataset
from eznlp.model import EncoderConfig
from eznlp.pretrained import BertLikeConfig
from eznlp.span_classification import SpanClassificationDecoderConfig
from eznlp.relation_classification import RelationClassificationDecoderConfig
from eznlp.relation_classification import JointClassifierConfig, JointTrainer


class TestClassifier(object):
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
        
        chunks_pred012, relations_pred012 = self.model.decode(batch012)
        chunks_pred123, relations_pred123 = self.model.decode(batch123)
        assert chunks_pred012[1:] == chunks_pred123[:-1]
        assert relations_pred012[1:] == relations_pred123[:-1]
        
        
    def _assert_trainable(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        trainer = JointTrainer(self.model, optimizer=optimizer, device=self.device)
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
    def test_classifier(self, enc_arch, agg_mode, conll2004_demo, device):
        self.config = JointClassifierConfig(intermediate2=EncoderConfig(arch=enc_arch), 
                                            ck_decoder=SpanClassificationDecoderConfig(agg_mode=agg_mode),
                                            rel_decoder=RelationClassificationDecoderConfig(agg_mode=agg_mode))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_classifier_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = JointClassifierConfig(ohots=None, 
                                            bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                            intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        