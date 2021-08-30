# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.training import Trainer
from eznlp.model import ImageEncoderConfig, GeneratorConfig, Image2TextConfig


class TestModel(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch = [self.dataset[i] for i in range(4)]
        batch012 = self.dataset.collate(batch[:3]).to(self.device)
        batch123 = self.dataset.collate(batch[1:]).to(self.device)
        losses012, states012 = self.model(batch012, return_states=True)
        losses123, states123 = self.model(batch123, return_states=True)
        
        logits012, logits123 = states012['logits'], states123['logits']
        min_step = min(logits012.size(1), logits123.size(1))
        delta_logits = logits012[1:, :min_step] - logits123[:-1, :min_step]
        assert delta_logits.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        pred012 = self.model.decode(batch012, **states012)
        pred123 = self.model.decode(batch123, **states123)
        pred012 = [yi[:min_step] for yi in pred012]
        pred123 = [yi[:min_step] for yi in pred123]
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
        
        
    @pytest.mark.parametrize("fl_gamma", [0.0, 2.0])
    def test_model(self, fl_gamma, flickr8k_demo_with_folder, resnet18_with_trans, device):
        flickr8k_demo, folder = flickr8k_demo_with_folder
        resnet, trans = resnet18_with_trans
        self.config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, transforms=trans, folder=folder), 
                                       decoder=GeneratorConfig(fl_gamma=fl_gamma))
        self._setup_case(flickr8k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, flickr8k_demo_with_folder, resnet18_with_trans, device):
        flickr8k_demo, folder = flickr8k_demo_with_folder
        resnet, trans = resnet18_with_trans
        self.config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, transforms=trans, folder=folder))
        self._setup_case(flickr8k_demo, device)
        
        data_wo_gold = [{'img_fn': entry['img_fn']} for entry in flickr8k_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        y_pred = trainer.predict(dataset_wo_gold)
        assert len(y_pred) == len(data_wo_gold)
