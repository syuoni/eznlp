# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, JointDecoderConfig, ModelConfig
from eznlp.model import SequenceTaggingDecoderConfig, SpanClassificationDecoderConfig, RelationClassificationDecoderConfig
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
        
        chunks_pred012, relations_pred012 = self.model.decode(batch012)
        chunks_pred123, relations_pred123 = self.model.decode(batch123)
        assert chunks_pred012[1:] == chunks_pred123[:-1]
        assert relations_pred012[1:] == relations_pred123[:-1]
        
        
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
    @pytest.mark.parametrize("ck_decoder", ['sequence_tagging', 'span_classification'])
    @pytest.mark.parametrize("agg_mode", ['max_pooling'])
    @pytest.mark.parametrize("criterion", ['cross_entropy', 'focal'])
    def test_model(self, arch, ck_decoder, agg_mode, criterion, conll2004_demo, device):
        if ck_decoder.lower() == 'sequence_tagging':
            ck_decoder_config = SequenceTaggingDecoderConfig(criterion=criterion)
        else:
            ck_decoder_config = SpanClassificationDecoderConfig(agg_mode=agg_mode, criterion=criterion)
        self.config = ModelConfig(intermediate2=EncoderConfig(arch=arch), 
                                  decoder=JointDecoderConfig(ck_decoder=ck_decoder_config, 
                                                             rel_decoder=RelationClassificationDecoderConfig(agg_mode=agg_mode)))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('joint', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, device):
        self.config = ModelConfig('joint')
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred, set_relations_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
        assert len(set_relations_pred) == len(dataset_wo_gold)

