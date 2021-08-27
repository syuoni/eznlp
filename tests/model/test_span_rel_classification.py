# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import BertLikeConfig, SpanRelClassificationDecoderConfig, JointExtractionDecoderConfig, ExtractorConfig
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
    @pytest.mark.parametrize("fl_gamma", [0.0, 2.0])
    def test_model(self, agg_mode, ck_label_emb_dim, fl_gamma, conll2004_demo, device):
        self.config = ExtractorConfig(decoder=SpanRelClassificationDecoderConfig(agg_mode=agg_mode, ck_label_emb_dim=ck_label_emb_dim, fl_gamma=fl_gamma))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ExtractorConfig('span_rel_classification', ohots=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                      intermediate2=None)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, device):
        self.config = ExtractorConfig('span_rel_classification')
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens'], 
                         'chunks': entry['chunks']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_relations_pred = trainer.predict(dataset_wo_gold)
        assert len(set_relations_pred) == len(data_wo_gold)



@pytest.mark.parametrize("num_neg_relations", [1, 100])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("building", [True, False])
def test_chunk_pairs_obj(EAR_data_demo, num_neg_relations, training, building):
    entry = EAR_data_demo[0]
    chunks, relations = entry['chunks'], entry['relations']
    
    if building:
        config = ExtractorConfig(decoder=SpanRelClassificationDecoderConfig(num_neg_relations=num_neg_relations))
        rel_decoder_config = config.decoder
    else:
        config = ExtractorConfig(decoder=JointExtractionDecoderConfig(attr_decoder=None, rel_decoder=SpanRelClassificationDecoderConfig(num_neg_relations=num_neg_relations)))
        rel_decoder_config = config.decoder.rel_decoder
    
    dataset = Dataset(EAR_data_demo, config, training=training)
    dataset.build_vocabs_and_dims()
    
    chunk_pairs_obj = dataset[0]['chunk_pairs_obj']
    assert chunk_pairs_obj.relations == relations
    
    if building:
        assert chunk_pairs_obj.chunks == chunks
        assert chunk_pairs_obj.is_built
    else:
        assert chunk_pairs_obj.chunks == chunks if training else len(chunk_pairs_obj.chunks) == 0
        assert not chunk_pairs_obj.is_built
        chunks_pred = [('EntA', 0, 1), ('EntB', 1, 2), ('EntA', 2, 3)]
        chunk_pairs_obj.inject_chunks(chunks_pred)
        chunk_pairs_obj.build(rel_decoder_config)
        assert len(chunk_pairs_obj.chunks) == len(chunks) + len(chunks_pred) if training else len(chunk_pairs_obj.chunks) == len(chunks_pred)
        assert chunk_pairs_obj.is_built
    
    # num_candidate_chunk_pairs = len(chunk_pairs_obj.chunks) * (len(chunk_pairs_obj.chunks) - 1)
    num_candidate_chunk_pairs = len([(head, tail) for head in chunk_pairs_obj.chunks for tail in chunk_pairs_obj.chunks 
                                         if head[1:] != tail[1:]
                                         and (head[0], tail[0]) in rel_decoder_config.legal_head_tail_types])
    
    rel_chunk_pairs = [(head, tail) for rel_label, head, tail in relations]
    oov_chunk_pairs = [(head, tail) for rel_label, head, tail in relations if not (head in chunk_pairs_obj.chunks and tail in chunk_pairs_obj.chunks)]
    if training:
        assert len(oov_chunk_pairs) == 0
        assert set(rel_chunk_pairs).issubset(set(chunk_pairs_obj.chunk_pairs))
        assert len(chunk_pairs_obj.chunk_pairs) == min(num_candidate_chunk_pairs, len(relations) + num_neg_relations)
    else:
        assert set(rel_chunk_pairs) - set(chunk_pairs_obj.chunk_pairs) == set(oov_chunk_pairs)
        assert len(chunk_pairs_obj.chunk_pairs) == num_candidate_chunk_pairs
    
    assert (chunk_pairs_obj.span_size_ids+1).tolist() == [[he-hs, te-ts] for (hl, hs, he), (tl, ts, te) in chunk_pairs_obj.chunk_pairs]
    
    chunk_pair2rel_label = {(head, tail): rel_label for rel_label, head, tail in relations}
    assert all(chunk_pair2rel_label.get(chunk_pair, rel_decoder_config.rel_none_label) == rel_decoder_config.idx2rel_label[rel_label_id]
                   for chunk_pair, rel_label_id
                   in zip(chunk_pairs_obj.chunk_pairs, chunk_pairs_obj.rel_label_ids.tolist()))
