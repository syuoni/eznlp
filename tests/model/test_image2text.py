# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import GenerationDataset
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
        
        self.dataset = GenerationDataset(data, self.config)
        self.dataset.build_vocabs_and_dims()
        self.model = self.config.instantiate().to(self.device)
        assert isinstance(self.config.name, str) and len(self.config.name) > 0
        
        
    @pytest.mark.parametrize("enc_arch", ['VGG', 'ResNet'])
    @pytest.mark.parametrize("dec_arch", ['LSTM', 'GRU', 'Gehring', 'Transformer'])
    def test_model(self, enc_arch, dec_arch, flickr8k_demo, vgg11_with_trans, resnet18_with_trans, device):
        if enc_arch.lower() == 'vgg':
            backbone, trans = vgg11_with_trans
        else:
            backbone, trans = resnet18_with_trans
        self.config = Image2TextConfig(encoder=ImageEncoderConfig(arch=enc_arch, backbone=backbone, transforms=trans), 
                                       decoder=GeneratorConfig(arch=dec_arch, use_emb2init_hid=True))
        self._setup_case(flickr8k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, flickr8k_demo, resnet18_with_trans, device):
        resnet, trans = resnet18_with_trans
        self.config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, transforms=trans))
        self._setup_case(flickr8k_demo, device)
        
        data_wo_gold = [{'img_path': entry['img_path']} for entry in flickr8k_demo]
        dataset_wo_gold = GenerationDataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        y_pred = trainer.predict(dataset_wo_gold)
        assert len(y_pred) == len(data_wo_gold)



@pytest.mark.parametrize("training", [True, False])
def test_generation_dataset(training, flickr8k_demo, resnet18_with_trans):
    resnet, trans = resnet18_with_trans
    config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, transforms=trans))
    
    dataset = GenerationDataset(flickr8k_demo, config, training=training)
    dataset.build_vocabs_and_dims()
    if training:
        assert len(dataset) == 10
        examples = [dataset[i] for i in range(len(dataset))]
        assert all((examples[i]['img']-examples[0]['img']).abs().max().item() < 1e-6 for i in range(5))
        assert all((examples[i]['img']-examples[0]['img']).abs().max().item() > 1e-4 for i in range(5, 10))
        assert len(set(tuple(ex['trg_tok_ids'].tolist()) for ex in examples)) == 10
    else:
        assert len(dataset) == 2
        examples = [dataset[i] for i in range(len(dataset))]
        assert (examples[1]['img'] - examples[0]['img']).abs().max().item() > 1e-4


@pytest.mark.parametrize("use_cache", [True, False])
def test_image_cache(use_cache, flickr8k_demo, resnet18_with_trans, device):
    resnet, trans = resnet18_with_trans
    config = Image2TextConfig(encoder=ImageEncoderConfig(backbone=resnet, transforms=trans, use_cache=use_cache))
    
    dataset = GenerationDataset(flickr8k_demo, config)
    dataset.build_vocabs_and_dims()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=dataset.collate)
    
    for batch in dataloader:
        batch = batch.to(device)
    if use_cache:
        assert all('img' in entry for entry in flickr8k_demo)
    else:
        assert all('img' not in entry for entry in flickr8k_demo)
