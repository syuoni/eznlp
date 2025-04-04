# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, SpanAttrClassificationDecoderConfig, ExtractorConfig
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
        for entry in data:
            entry['chunks_pred'] = entry['chunks']
        self.device = device

        self.dataset = Dataset(data, self.config)
        self.dataset.build_vocabs_and_dims()
        self.model = self.config.instantiate().to(self.device)
        assert isinstance(self.config.name, str) and len(self.config.name) > 0


    @pytest.mark.parametrize("red_dim", [100, 0])
    @pytest.mark.parametrize("ck_loss_weight", [0, 0.5])
    @pytest.mark.parametrize("multilabel", [True, False])
    def test_model(self, red_dim, ck_loss_weight, multilabel, HwaMei_demo, device):
        self.config = ExtractorConfig(decoder=SpanAttrClassificationDecoderConfig(reduction=EncoderConfig(arch='FFN', hid_dim=red_dim, num_layers=1, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0),
                                                                                  ck_loss_weight=ck_loss_weight, multilabel=multilabel))
        self._setup_case(HwaMei_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()


    def test_model_with_bert_like(self, HwaMei_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ExtractorConfig('span_attr_classification', ohots=None,
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert),
                                      intermediate2=None)
        self._setup_case(HwaMei_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()


    def test_prediction_without_gold(self, HwaMei_demo, device):
        self.config = ExtractorConfig('span_attr_classification')
        self._setup_case(HwaMei_demo, device)

        data_wo_gold = [{'tokens': entry['tokens'],
                         'chunks_pred': entry['chunks']} for entry in HwaMei_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)

        trainer = Trainer(self.model, device=device)
        set_attributes_pred = trainer.predict(dataset_wo_gold)
        assert len(set_attributes_pred) == len(data_wo_gold)
