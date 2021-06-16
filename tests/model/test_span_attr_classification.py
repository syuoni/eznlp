# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, SpanAttrClassificationDecoderConfig, ModelConfig
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
        
        
    @pytest.mark.parametrize("agg_mode", ['max_pooling', 'multiplicative_attention'])
    @pytest.mark.parametrize("ck_label_emb_dim", [25, 0])
    @pytest.mark.parametrize("criterion", ['BCE'])
    def test_model(self, agg_mode, ck_label_emb_dim, criterion, HwaMei_demo, device):
        self.config = ModelConfig(decoder=SpanAttrClassificationDecoderConfig(agg_mode=agg_mode, ck_label_emb_dim=ck_label_emb_dim, criterion=criterion))
        self._setup_case(HwaMei_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, HwaMei_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('span_attr_classification', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                  intermediate2=None)
        self._setup_case(HwaMei_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, HwaMei_demo, device):
        self.config = ModelConfig('span_attr_classification')
        self._setup_case(HwaMei_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens'], 
                         'chunks': entry['chunks']} for entry in HwaMei_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_attributes_pred = trainer.predict(dataset_wo_gold)
        assert len(set_attributes_pred) == len(data_wo_gold)



# TODO
@pytest.mark.parametrize("building", [True, False])
def test_chunks_obj(EAR_data_demo, building):
    entry = EAR_data_demo[0]
    chunks, attributes = entry['chunks'], entry['attributes']

    config = ModelConfig(decoder=SpanAttrClassificationDecoderConfig())
    attr_decoder_config = config.decoder

    dataset = Dataset(EAR_data_demo, config, training=True)
    dataset.build_vocabs_and_dims()

    chunks_obj = dataset[0]['chunks_obj']
    assert chunks_obj.attributes == attributes

    assert chunks_obj.chunks == chunks
    assert chunks_obj.is_built

    assert (chunks_obj.span_size_ids+1).tolist() == [e-s for l, s, e in chunks_obj.chunks]
    assert chunks_obj.attr_label_ids.size(0) == len(chunks_obj.chunks)
    
    attributes_retr = []
    for chunk, attr_label_ids in zip(chunks_obj.chunks, chunks_obj.attr_label_ids):
        attr_labels = [attr_decoder_config.idx2attr_label[i] for i, l in enumerate(attr_label_ids.tolist()) if l > 0]
        if attr_decoder_config.attr_none_label not in attr_labels:
            attributes_retr.extend([(attr_label, chunk) for attr_label in attr_labels])
    assert set(attributes_retr) == set(attributes)
