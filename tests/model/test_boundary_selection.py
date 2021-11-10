# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, BoundarySelectionDecoderConfig, ExtractorConfig
from eznlp.model.decoder.boundary_selection import _spans_from_upper_triangular
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
        
        
    def test_model_with_bert_like(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = ExtractorConfig('boundary_selection', 
                                      ohots=None, 
                                      bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert), 
                                      intermediate2=None)
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



@pytest.mark.parametrize("sb_epsilon", [0.0, 0.1])
@pytest.mark.parametrize("sl_epsilon", [0.0, 0.1])
def test_boundaries_obj(sb_epsilon, sl_epsilon, EAR_data_demo):
    entry = EAR_data_demo[0]
    tokens, chunks = entry['tokens'], entry['chunks']
    
    config = ExtractorConfig(decoder=BoundarySelectionDecoderConfig(sb_epsilon=sb_epsilon, sl_epsilon=sl_epsilon))
    dataset = Dataset(EAR_data_demo, config)
    dataset.build_vocabs_and_dims()
    boundaries_obj = dataset[0]['boundaries_obj']
    
    num_tokens, num_chunks = len(tokens), len(chunks)
    assert boundaries_obj.chunks == chunks
    if sb_epsilon == 0 and sl_epsilon == 0:
        assert all(boundaries_obj.boundary2label_id[start, end-1] == config.decoder.label2idx[label] for label, start, end in chunks)
        labels_retr = [config.decoder.idx2label[i] for i in boundaries_obj.boundary2label_id[torch.arange(num_tokens) >= torch.arange(num_tokens).unsqueeze(-1)].tolist()]
    else:
        assert all(boundaries_obj.boundary2label_id[start, end-1].argmax() == config.decoder.label2idx[label] for label, start, end in chunks)
        labels_retr = [config.decoder.idx2label[i] for i in boundaries_obj.boundary2label_id[torch.arange(num_tokens) >= torch.arange(num_tokens).unsqueeze(-1)].argmax(dim=-1).tolist()]
        
        assert (boundaries_obj.boundary2label_id.sum(dim=-1) - 1).abs().max().item() < 1e-6
        if sb_epsilon == 0:
            assert (boundaries_obj.boundary2label_id[:, :, config.decoder.none_idx] < 1).sum().item() == num_chunks
        else:
            assert (boundaries_obj.boundary2label_id[:, :, config.decoder.none_idx] < 1).sum().item() > num_chunks
    
    chunks_retr = [(label, start, end) for label, (start, end) in zip(labels_retr, _spans_from_upper_triangular(num_tokens)) if label != config.decoder.none_label]
    assert set(chunks_retr) == set(chunks)
    
    extractor = config.instantiate()
    # \sum_{k=0}^N k(N-k), where N is `num_tokens`
    assert extractor.decoder._span_size_ids.sum().item() == (num_tokens**3 - num_tokens) // 6
    assert extractor.decoder._span_non_mask.sum().item() == (num_tokens**2 + num_tokens) // 2



@pytest.mark.parametrize("sb_epsilon", [0.1, 0.5])
@pytest.mark.parametrize("sb_size", [1, 2, 3])
def test_boundaries_obj_for_boundary_smoothing(sb_epsilon, sb_size):
    entry = {'tokens': list("abcdef"), 
             'chunks': [('EntA', 0, 1), ('EntA', 0, 4), ('EntB', 0, 5), ('EntA', 3, 5), ('EntA', 4, 5)]}
    config = BoundarySelectionDecoderConfig(sb_epsilon=sb_epsilon, sb_size=sb_size)
    config.build_vocab([entry])
    boundaries_obj = config.exemplify(entry)['boundaries_obj']
    
    num_tokens, num_chunks = len(entry['tokens']), len(entry['chunks'])
    span_sizes = torch.arange(num_tokens) - torch.arange(num_tokens).unsqueeze(-1) + 1
    assert (boundaries_obj.boundary2label_id.sum(dim=-1) - 1).abs().max().item() < 1e-6
    assert (boundaries_obj.boundary2label_id[:, :, 1:].sum() - num_chunks).abs().max().item() < 1e-6
    assert (boundaries_obj.boundary2label_id[span_sizes<=0] - torch.tensor([1.0, 0.0, 0.0])).abs().max().item() < 1e-6
    
    if sb_size == 1:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(1/2)*sb_epsilon, 1-(3/4)*sb_epsilon, (1/4)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(1/2)*sb_epsilon, (1/4)*sb_epsilon, 1-(3/4)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(3/4)*sb_epsilon, 1-(3/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
    elif sb_size == 2:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(1/4)*sb_epsilon, 1-(1/4)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(9/16)*sb_epsilon, 1-(11/16)*sb_epsilon, (1/8)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(1/2)*sb_epsilon, (1/8)*sb_epsilon, 1-(5/8)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(5/8)*sb_epsilon, 1-(5/8)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(3/8)*sb_epsilon, 1-(3/8)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
    elif sb_size == 3:
        assert (boundaries_obj.boundary2label_id[0, 0] - torch.tensor([(7/36)*sb_epsilon, 1-(7/36)*sb_epsilon, 0.0])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 3] - torch.tensor([(37/72)*sb_epsilon, 1-(43/72)*sb_epsilon, (1/12)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[0, 4] - torch.tensor([(4/9)*sb_epsilon, (1/9)*sb_epsilon, 1-(5/9)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[3, 4] - torch.tensor([(19/36)*sb_epsilon, 1-(5/9)*sb_epsilon, (1/36)*sb_epsilon])).abs().max().item() < 1e-6
        assert (boundaries_obj.boundary2label_id[4, 4] - torch.tensor([(1/3)*sb_epsilon, 1-(1/3)*sb_epsilon, 0.0])).abs().max().item() < 1e-6



@pytest.mark.parametrize("neg_sampling_rate, hard_neg_sampling_rate", [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), 
                                                                       (0.3, 0.6), (0.1, 0.9), (0.2, 0.8)])
@pytest.mark.parametrize("training", [True, False])
def test_boundaries_obj_for_neg_sampling(neg_sampling_rate, hard_neg_sampling_rate, training):
    entry = {'tokens': list("abcdefhijk"), 
             'chunks': [('EntA', 0, 1), ('EntA', 0, 4), ('EntB', 0, 5), ('EntA', 3, 5), ('EntA', 4, 5)]}
    config = BoundarySelectionDecoderConfig(neg_sampling_rate=neg_sampling_rate, 
                                            hard_neg_sampling_rate=hard_neg_sampling_rate, 
                                            hard_neg_sampling_size=3)
    config.build_vocab([entry])
    boundaries_obj = config.exemplify(entry, training=training)['boundaries_obj']
    
    if (not training) or (neg_sampling_rate >= 1):
        assert not hasattr(boundaries_obj, 'non_mask')
    elif neg_sampling_rate <=0 and hard_neg_sampling_rate <= 0:
        assert boundaries_obj.non_mask.sum().item() == 5
    elif neg_sampling_rate <=0 and hard_neg_sampling_rate >= 1:
        assert boundaries_obj.non_mask.sum().item() == 30
    else:
        assert abs(boundaries_obj.non_mask.sum().item() - (25*neg_sampling_rate + 25*hard_neg_sampling_rate + 5)) < 5


@pytest.mark.parametrize("seq_len", [1, 5, 10, 100])
def test_spans_from_upper_triangular(seq_len):
    assert len(list(_spans_from_upper_triangular(seq_len))) == (seq_len+1)*seq_len // 2
