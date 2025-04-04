# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, SpanBertLikeConfig
from eznlp.model import SpecificSpanRelClsDecoderConfig, UnfilteredSpecificSpanRelClsDecoderConfig
from eznlp.model import SpecificSpanExtractorConfig
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
        for k in range(self.config.decoder.min_span_size, self.config.decoder.max_span_size+1):
            query012, query123 = states012['all_query_hidden'][k], states123['all_query_hidden'][k]
            delta_query = query012[1:, :min_step-k+1] - query123[:-1, :min_step-k+1]
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


    @pytest.mark.slow
    @pytest.mark.parametrize("use_unfiltered, num_layers, min_span_size, use_context, context_mode, fusing_mode, ck_loss_weight",
                             [(False, 3,  2, True,  'specific', 'affine', 0.0),  # Baseline
                              (False, 12, 2, True,  'specific', 'affine', 0.0),  # Number of layers
                              (False, 3,  1, True,  'specific', 'affine', 0.0),  # Minimum span size
                              (False, 3,  2, False, 'specific', 'affine', 0.0),  # Context
                              (False, 3,  2, True,  'shallow',  'affine', 0.0),
                              (False, 3,  2, True,  'specific', 'concat', 0.0),  # Fusing mode
                              (False, 3,  2, True,  'specific', 'affine', 0.5),  # Chunk loss weight
                              (True,  3,  2, True,  'specific', 'affine', 0.0),  # Unfiltered chunk
                              (True,  3,  1, True,  'specific', 'affine', 0.0),
                              (True,  3,  2, True,  'specific', 'concat', 0.0)])
    def test_model(self, use_unfiltered, num_layers, min_span_size, use_context, context_mode, fusing_mode, ck_loss_weight, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        if use_unfiltered:
            decoder_config = UnfilteredSpecificSpanRelClsDecoderConfig(fusing_mode=fusing_mode, min_span_size=min_span_size, max_span_size=3)
        else:
            decoder_config = SpecificSpanRelClsDecoderConfig(use_context=use_context, context_mode=context_mode, fusing_mode=fusing_mode, ck_loss_weight=ck_loss_weight, min_span_size=min_span_size, max_span_size=3)
        self.config = SpecificSpanExtractorConfig(decoder=decoder_config,
                                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True),
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert, freeze=False, num_layers=num_layers, share_weights_ext=True, share_weights_int=True),
                                                  intermediate2=EncoderConfig(arch='LSTM', hid_dim=200))
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()


    @pytest.mark.parametrize("use_unfiltered", [False, True])
    def test_prediction_without_gold(self, use_unfiltered, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(decoder='unfiltered_specific_span_rel' if use_unfiltered else 'specific_span_rel',
                                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True),
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert),
                                                  intermediate2=None)
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)

        data_wo_gold = [{'tokens': entry['tokens'],
                         'chunks_pred': entry['chunks']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)

        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
