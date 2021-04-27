# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.token import TokenSequence
from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, RelationClassificationDecoderConfig, ModelConfig
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
        
        
    @pytest.mark.parametrize("enc_arch", ['Conv', 'LSTM'])
    @pytest.mark.parametrize("agg_mode", ['max_pooling'])
    def test_model(self, enc_arch, agg_mode, conll2004_demo, device):
        self.config = ModelConfig(intermediate2=EncoderConfig(arch=enc_arch), 
                                  decoder=RelationClassificationDecoderConfig(agg_mode=agg_mode))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ModelConfig('relation_classification', 
                                  ohots=None, 
                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_labels(self, conll2004_demo, device):
        self.config = ModelConfig('relation_classification')
        self._setup_case(conll2004_demo, device)
        
        data_wo_labels = [{'tokens': entry['tokens'], 
                           'chunks': entry['chunks']} for entry in conll2004_demo]
        dataset_wo_labels = Dataset(data_wo_labels, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_labels_pred = trainer.predict(dataset_wo_labels)
        assert len(set_labels_pred) == len(data_wo_labels)




@pytest.mark.parametrize("num_neg_relations, training", [(1, True), 
                                                         (1, False)])
def test_span_pairs_obj(num_neg_relations, training):
    tokenized_raw_text = ["This", "is", "a", "-3.14", "demo", ".", 
                          "Those", "are", "an", "APPLE", "and", "some", "glass", "bottles", "."]
    tokens = TokenSequence.from_tokenized_text(tokenized_raw_text)
    
    entities = [{'type': 'EntA', 'start': 4, 'end': 5},
                {'type': 'EntA', 'start': 9, 'end': 10},
                {'type': 'EntB', 'start': 12, 'end': 14}]
    chunks = [(ent['type'], ent['start'], ent['end']) for ent in entities]
    
    raw_relations = [{'type': 'RelA', 'head': 0, 'tail': 1}, 
                     {'type': 'RelA', 'head': 0, 'tail': 2}, 
                     {'type': 'RelB', 'head': 1, 'tail': 2}, 
                     {'type': 'RelB', 'head': 2, 'tail': 1}]
    relations = [(rel['type'], chunks[rel['head']], chunks[rel['tail']]) for rel in raw_relations]
    data = [{'tokens': tokens, 'chunks': chunks, 'relations': relations}]
    
    config = ModelConfig(decoder=RelationClassificationDecoderConfig(num_neg_relations=num_neg_relations))
    dataset = Dataset(data, config, training=training)
    dataset.build_vocabs_and_dims()
    
    span_pairs_obj = dataset[0]['span_pairs_obj']
    assert span_pairs_obj.relations == relations
    assert set((rel[1][1], rel[1][2], rel[2][1], rel[2][2]) for rel in relations).issubset(set(span_pairs_obj.sp_pairs))
    assert (span_pairs_obj.sp_pair_size_ids+1).tolist() == [[he-hs, te-ts] for hs, he, ts, te in span_pairs_obj.sp_pairs]
    
    num_chunks = len(chunks)
    expected_num_sp_pairs = num_chunks * (num_chunks-1)
    if training:
        expected_num_sp_pairs = min(expected_num_sp_pairs, len(relations) + num_neg_relations)
    assert len(span_pairs_obj.sp_pairs) == expected_num_sp_pairs
    
    assert (span_pairs_obj.rel_label_ids[:len(relations)] != config.decoder.rel_none_idx).all().item()
    assert (span_pairs_obj.rel_label_ids[len(relations):] == config.decoder.rel_none_idx).all().item()
