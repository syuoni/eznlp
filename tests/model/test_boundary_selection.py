# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, BoundarySelectionDecoderConfig, ExtractorConfig
from eznlp.model import BertLikePreProcessor
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
        assert delta_losses.abs().max().item() < 5e-4
        
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
        
        
    @pytest.mark.parametrize("use_biaffine", [True, False])
    @pytest.mark.parametrize("affine_arch", ['FFN', 'LSTM'])
    @pytest.mark.parametrize("size_emb_dim", [25, 0])
    @pytest.mark.parametrize("fl_gamma, sl_epsilon, sb_epsilon, sb_size", [(0.0, 0.0, 0.0, 1), 
                                                                           (2.0, 0.0, 0.0, 1), 
                                                                           (0.0, 0.1, 0.0, 1), 
                                                                           (0.0, 0.0, 0.1, 1), 
                                                                           (0.0, 0.0, 0.1, 2), 
                                                                           (0.0, 0.0, 0.1, 3), 
                                                                           (0.0, 0.1, 0.1, 1)])
    def test_model(self, use_biaffine, affine_arch, size_emb_dim, fl_gamma, sl_epsilon, sb_epsilon, sb_size, conll2004_demo, device):
        self.config = ExtractorConfig(decoder=BoundarySelectionDecoderConfig(use_biaffine=use_biaffine, 
                                                                             affine=EncoderConfig(arch=affine_arch), 
                                                                             size_emb_dim=size_emb_dim, 
                                                                             fl_gamma=fl_gamma, sl_epsilon=sl_epsilon, sb_epsilon=sb_epsilon, sb_size=sb_size))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_wo_prod(self, conll2004_demo, device):
        self.config = ExtractorConfig(decoder=BoundarySelectionDecoderConfig(use_prod=False))
        self._setup_case(conll2004_demo, device)
        assert not hasattr(self.model.decoder, 'U')
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("from_subtokenized", [False, True])
    def test_model_with_bert_like(self, from_subtokenized, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ExtractorConfig('boundary_selection', ohots=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, from_subtokenized=from_subtokenized), 
                                      intermediate2=None)
        if from_subtokenized:
            preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
            conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, device):
        self.config = ExtractorConfig('boundary_selection')
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
