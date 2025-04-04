# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, MaskedSpanBertLikeConfig
from eznlp.model import MaskedSpanAttrClsDecoderConfig
from eznlp.model import MaskedSpanExtractorConfig
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

        # Check the consistency of query hidden states
        for key in ['span_query_hidden']:
            query012, query123 = states012[key], states123[key]
            min_step = min(query012.size(1), query123.size(1))
            delta_query = query012[1:, :min_step] - query123[:-1, :min_step]
            assert delta_query.abs().max().item() < 1e-4

        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 5e-3

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


    @pytest.mark.parametrize("num_layers, use_init_size_emb, red_dim, ck_loss_weight",
                             [(3,  False, 100, 0.0),  # Baseline
                              (12, False, 100, 0.0),  # Number of layers
                              (3,  True,  100, 0.0),  # Use initial size embedding
                              (3,  False, 100, 0.1)]) # Chunk loss weight
    def test_model(self, num_layers, use_init_size_emb, red_dim, ck_loss_weight, HwaMei_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        decoder_config = MaskedSpanAttrClsDecoderConfig(reduction=EncoderConfig(arch='FFN', hid_dim=red_dim, num_layers=1, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0),
                                                        ck_loss_weight=ck_loss_weight)
        self.config = MaskedSpanExtractorConfig(decoder=decoder_config,
                                                bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True),
                                                masked_span_bert_like=MaskedSpanBertLikeConfig(bert_like=bert, freeze=False, num_layers=num_layers, use_init_size_emb=use_init_size_emb, share_weights_ext=True, share_weights_int=True))
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        HwaMei_demo = preprocessor.subtokenize_for_data(HwaMei_demo)
        self._setup_case(HwaMei_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()


    def test_prediction_without_gold(self, HwaMei_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = MaskedSpanExtractorConfig(decoder='masked_span_attr',
                                                bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True),
                                                masked_span_bert_like=MaskedSpanBertLikeConfig(bert_like=bert))
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        HwaMei_demo = preprocessor.subtokenize_for_data(HwaMei_demo)
        self._setup_case(HwaMei_demo, device)

        data_wo_gold = [{'tokens': entry['tokens'],
                         'chunks_pred': entry['chunks']} for entry in HwaMei_demo]
        data_wo_gold = preprocessor.subtokenize_for_data(data_wo_gold)
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)

        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
