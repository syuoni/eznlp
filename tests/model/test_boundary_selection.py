# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, BoundarySelectionDecoderConfig, ModelConfig
from eznlp.model.decoder.boundary_selection import _generate_spans_from_upper_triangular
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
        assert delta_losses.abs().max().item() < 5e-4
        
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
        
        
    @pytest.mark.parametrize("use_biaffine", [True, False])
    @pytest.mark.parametrize("affine_arch", ['FFN', 'LSTM'])
    @pytest.mark.parametrize("size_emb_dim", [25, 0])
    @pytest.mark.parametrize("fl_gamma, sl_epsilon, sb_epsilon", [(0.0, 0.0, 0.0), 
                                                                  (2.0, 0.0, 0.0), 
                                                                  (0.0, 0.1, 0.0), 
                                                                  (0.0, 0.0, 0.1), 
                                                                  (0.0, 0.1, 0.1)])
    def test_model(self, use_biaffine, affine_arch, size_emb_dim, fl_gamma, sl_epsilon, sb_epsilon, conll2004_demo, device):
        self.config = ModelConfig(decoder=BoundarySelectionDecoderConfig(use_biaffine=use_biaffine, 
                                                                         affine=EncoderConfig(arch=affine_arch), 
                                                                         size_emb_dim=size_emb_dim, 
                                                                         fl_gamma=fl_gamma, sl_epsilon=sl_epsilon, sb_epsilon=sb_epsilon))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('boundary_selection', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, device):
        self.config = ModelConfig('boundary_selection')
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)



def test_boundaries_obj(EAR_data_demo):
    entry = EAR_data_demo[0]
    tokens, chunks = entry['tokens'], entry['chunks']
    
    config = ModelConfig('boundary_selection')
    dataset = Dataset(EAR_data_demo, config)
    dataset.build_vocabs_and_dims()
    
    boundaries_obj = dataset[0]['boundaries_obj']
    assert boundaries_obj.chunks == chunks
    assert all(boundaries_obj.boundary2label_id[start, end-1] == config.decoder.label2idx[label] for label, start, end in chunks)
    
    labels_retr = [config.decoder.idx2label[i] for i in boundaries_obj.boundary2label_id[torch.arange(len(tokens)) >= torch.arange(len(tokens)).unsqueeze(-1)].tolist()]
    chunks_retr = [(label, start, end) for label, (start, end) in zip(labels_retr, _generate_spans_from_upper_triangular(len(tokens))) if label != config.decoder.none_label]
    assert set(chunks_retr) == set(chunks)



@pytest.mark.parametrize("seq_len", [1, 5, 10, 100])
def test_generate_spans_from_upper_triangular(seq_len):
    assert len(list(_generate_spans_from_upper_triangular(seq_len))) == (seq_len+1)*seq_len // 2
