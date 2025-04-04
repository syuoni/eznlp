# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import (BertLikeConfig, BertLikePreProcessor, EncoderConfig,
                         SpanBertLikeConfig, SpecificSpanClsDecoderConfig,
                         SpecificSpanExtractorConfig)
from eznlp.training import Trainer, count_params


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


    @pytest.mark.slow
    @pytest.mark.parametrize("from_subtokenized, num_layers, use_init_size_emb, min_span_size, agg_mode, share_weights_ext, share_weights_int, use_interm2, share_interm2, neg_sampling_rate, size_emb_dim, sb_epsilon",
                             [(False, 3,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),  # Baseline
                              (True,  3,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),  # Pre-subtokenization
                              (False, 12, False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),  # Number of layers
                              (False, 6,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),
                              (False, 1,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),
                              (False, 3,  True,  2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),  # Use initial size embedding
                              (False, 3,  False, 1, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0),  # Minimum span size
                              (False, 3,  False, 2, 'mean_pooling', True,  True,  False, True,  1.0, 0, 0),  # Initial aggregation
                              (False, 3,  False, 2, 'multiplicative_attention', True,  True,  False, True,  1.0, 0, 0),
                              (False, 3,  False, 2, 'conv',         True,  True,  False, True,  1.0, 0, 0),
                              (False, 3,  False, 1, 'conv',         True,  True,  False, True,  1.0, 0, 0),
                              (False, 3,  False, 2, 'max_pooling',  False, True,  False, True,  1.0, 0, 0),  # Share weights
                              (False, 3,  False, 2, 'max_pooling',  False, False, False, True,  1.0, 0, 0),
                              (False, 3,  False, 2, 'max_pooling',  True,  True,  True,  True,  1.0, 0, 0),  # Use interm2
                              (False, 3,  False, 2, 'max_pooling',  False, True,  True,  True,  1.0, 0, 0),  # Use interm2 x Share weights x Share interm2
                              (False, 3,  False, 2, 'max_pooling',  False, False, True,  True,  1.0, 0, 0),
                              (False, 3,  False, 2, 'max_pooling',  False, True,  True,  False, 1.0, 0, 0),
                              (False, 3,  False, 2, 'max_pooling',  False, False, True,  False, 1.0, 0, 0),
                              (False, 3,  False, 2, 'max_pooling',  True,  True,  False, True,  0.5, 0, 0),  # Use negative sampling
                              (False, 3,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 25, 0),  # Use size embedding
                              (False, 3,  False, 2, 'max_pooling',  True,  True,  False, True,  1.0, 0, 0.1)])  # Use boundary smoothing
    def test_model(self, from_subtokenized, num_layers, use_init_size_emb, min_span_size, agg_mode, share_weights_ext, share_weights_int, use_interm2, share_interm2, neg_sampling_rate, size_emb_dim, sb_epsilon, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(decoder=SpecificSpanClsDecoderConfig(neg_sampling_rate=neg_sampling_rate, size_emb_dim=size_emb_dim, min_span_size=min_span_size, max_span_size=3, sb_epsilon=sb_epsilon),
                                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, from_subtokenized=from_subtokenized, freeze=False, output_hidden_states=True),
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert, freeze=False, num_layers=num_layers, use_init_size_emb=use_init_size_emb, share_weights_ext=share_weights_ext, share_weights_int=share_weights_int, init_agg_mode=agg_mode),
                                                  intermediate2=EncoderConfig(arch='LSTM', hid_dim=400) if use_interm2 else None,
                                                  share_interm2=share_interm2)
        if from_subtokenized:
            preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
            conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)

        num_model_params = count_params(self.model)
        num_query_params = count_params(self.model.span_bert_like.query_bert_like)
        num_other_params = (count_params(self.model.decoder, verbose=False) +
                            count_params(self.model.bert_like, verbose=False) +
                            count_params(self.model.span_bert_like.init_aggregating, verbose=False) +
                            count_params(getattr(self.model.span_bert_like, 'size_embedding', []), verbose=False) +
                            count_params(getattr(self.model, 'intermediate2', []), verbose=False) +
                            count_params(getattr(self.model, 'span_intermediate2', []), verbose=False))
        if share_weights_ext:
            assert num_model_params == num_other_params
        else:
            assert num_model_params == num_other_params + num_query_params

        if share_weights_int:
            assert len(self.model.span_bert_like.query_bert_like.layer) == num_layers
            assert num_query_params == count_params(bert.encoder.layer[0], verbose=False) * num_layers
        else:
            assert len(self.model.span_bert_like.query_bert_like[0].layer) == num_layers
            assert num_query_params == count_params(bert.encoder.layer[0], verbose=False) * num_layers * 2  # `max_span_size` is 3

        self._assert_batch_consistency()
        self._assert_trainable()


    @pytest.mark.parametrize("inex_mkmmd_lambda", [0.0, 0.5, 1.0])
    def test_model_with_inex_mkmmd(self, inex_mkmmd_lambda, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(decoder=SpecificSpanClsDecoderConfig(inex_mkmmd_lambda=inex_mkmmd_lambda, nested_sampling_rate=0.0, max_span_size=3),
                                                  bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, output_hidden_states=True),
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert, num_layers=2))
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()


    def test_prediction_without_gold(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = SpecificSpanExtractorConfig(bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, output_hidden_states=True),
                                                  span_bert_like=SpanBertLikeConfig(bert_like=bert, num_layers=2),
                                                  intermediate2=None)
        self._setup_case(conll2004_demo, device)

        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2004_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)

        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
