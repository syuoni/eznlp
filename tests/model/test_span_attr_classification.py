# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import BertLikeConfig, SpanAttrClassificationDecoderConfig, JointExtractionDecoderConfig, ExtractorConfig
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
    @pytest.mark.parametrize("ck_label_emb_dim", [25, 0])
    def test_model(self, agg_mode, ck_label_emb_dim, HwaMei_demo, device):
        self.config = ExtractorConfig(decoder=SpanAttrClassificationDecoderConfig(agg_mode=agg_mode, ck_label_emb_dim=ck_label_emb_dim))
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
                         'chunks': entry['chunks']} for entry in HwaMei_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_attributes_pred = trainer.predict(dataset_wo_gold)
        assert len(set_attributes_pred) == len(data_wo_gold)



@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("building", [True, False])
def test_chunks_obj(EAR_data_demo, training, building):
    entry = EAR_data_demo[0]
    chunks, attributes = entry['chunks'], entry['attributes']
    
    if building:
        config = ExtractorConfig(decoder=SpanAttrClassificationDecoderConfig())
        attr_decoder_config = config.decoder
    else:
        config = ExtractorConfig(decoder=JointExtractionDecoderConfig(attr_decoder=SpanAttrClassificationDecoderConfig(), rel_decoder=None))
        attr_decoder_config = config.decoder.attr_decoder
    
    dataset = Dataset(EAR_data_demo, config, training=training)
    dataset.build_vocabs_and_dims()
    
    chunks_obj = dataset[0]['chunks_obj']
    assert chunks_obj.attributes == attributes
    
    if building:
        assert chunks_obj.chunks == chunks
        assert chunks_obj.is_built
    else:
        assert chunks_obj.chunks == chunks if training else len(chunks_obj.chunks) == 0
        assert not chunks_obj.is_built
        chunks_pred = [('EntA', 0, 1), ('EntB', 1, 2), ('EntA', 2, 3)]
        chunks_obj.inject_chunks(chunks_pred)
        chunks_obj.build(attr_decoder_config)
        assert len(chunks_obj.chunks) == len(chunks) + len(chunks_pred) if training else len(chunks_obj.chunks) == len(chunks_pred)
        assert chunks_obj.is_built
    
    assert (chunks_obj.span_size_ids+1).tolist() == [e-s for l, s, e in chunks_obj.chunks]
    assert chunks_obj.attr_label_ids.size(0) == len(chunks_obj.chunks)
    
    attributes_retr = []
    for chunk, attr_label_ids in zip(chunks_obj.chunks, chunks_obj.attr_label_ids):
        attr_labels = [attr_decoder_config.idx2attr_label[i] for i, l in enumerate(attr_label_ids.tolist()) if l > 0]
        if attr_decoder_config.attr_none_label not in attr_labels:
            attributes_retr.extend([(attr_label, chunk) for attr_label in attr_labels])
    if training or building:
        assert set(attributes_retr) == set(attributes)
    else:
        assert len(attributes_retr) == 0
