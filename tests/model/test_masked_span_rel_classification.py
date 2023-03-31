# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import Dataset
from eznlp.model import EncoderConfig, BertLikeConfig, MaskedSpanBertLikeConfig, SpanBertLikeConfig
from eznlp.model import MaskedSpanRelClsDecoderConfig, SpecificSpanClsDecoderConfig
from eznlp.model import MaskedSpanExtractorConfig, SpecificSpanExtractorConfig
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
        for key in ['span_query_hidden', 'ctx_query_hidden']: 
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
        
        
    @pytest.mark.parametrize("num_layers, use_context, context_mode, context_ext_win, context_exc_ck, fusing_mode, ck_loss_weight, use_inv_rel", 
                             [(3,  True,  'pair-specific', 0, True,  'affine', 0.0, False),  # Baseline
                              (12, True,  'pair-specific', 0, True,  'affine', 0.0, False),  # Number of layers
                              (3,  False, 'pair-specific', 0, True,  'affine', 0.0, False),  # Context
                              (3,  True,  'specific',      5, True,  'affine', 0.0, False), 
                              (3,  True,  'pair-specific', 5, True,  'affine', 0.0, False), 
                              (3,  True,  'pair-specific', 0, False, 'affine', 0.0, False), 
                              (3,  True,  'pair-specific', 0, True,  'concat', 0.0, False),  # Fusing mode
                              (3,  True,  'specific',      5, True,  'concat', 0.0, False), 
                              (3,  True,  'pair-specific', 0, True,  'affine', 0.5, False),  # Chunk loss weight
                              (3,  True,  'pair-specific', 0, True,  'affine', 0.0, True),   # Inverse relation
                              (3,  True,  'specific',      5, True,  'affine', 0.0, True)]) 
    def test_model(self, num_layers, use_context, context_mode, context_ext_win, context_exc_ck, fusing_mode, ck_loss_weight, use_inv_rel, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        decoder_config = MaskedSpanRelClsDecoderConfig(use_context=use_context, context_mode=context_mode, context_ext_win=context_ext_win, context_exc_ck=context_exc_ck, fusing_mode=fusing_mode, ck_loss_weight=ck_loss_weight, use_inv_rel=use_inv_rel)
        self.config = MaskedSpanExtractorConfig(decoder=decoder_config, 
                                                bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True), 
                                                masked_span_bert_like=MaskedSpanBertLikeConfig(bert_like=bert, freeze=False, num_layers=num_layers, share_weights_ext=True, share_weights_int=True))
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2004_demo, bert_with_tokenizer, device):
        bert, tokenizer = bert_with_tokenizer
        self.config = MaskedSpanExtractorConfig(decoder='masked_span_rel', 
                                                bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True), 
                                                masked_span_bert_like=MaskedSpanBertLikeConfig(bert_like=bert))
        preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
        conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
        self._setup_case(conll2004_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens'], 
                         'chunks_pred': entry['chunks']} for entry in conll2004_demo]
        data_wo_gold = preprocessor.subtokenize_for_data(data_wo_gold)
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)



@pytest.mark.parametrize("min_span_size", [2, 1])
def test_consistency_with_span_bert_like(min_span_size, conll2004_demo, bert_with_tokenizer, device):
    bert, tokenizer = bert_with_tokenizer
    
    sse_config = SpecificSpanExtractorConfig(decoder=SpecificSpanClsDecoderConfig(min_span_size=min_span_size, max_span_size=5), 
                                             bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True), 
                                             span_bert_like=SpanBertLikeConfig(bert_like=bert), 
                                             intermediate2=None)
    preprocessor = BertLikePreProcessor(tokenizer, verbose=False)
    conll2004_demo = preprocessor.subtokenize_for_data(conll2004_demo)
    sse_dataset = Dataset(conll2004_demo, sse_config)
    sse_dataset.build_vocabs_and_dims()
    sse_model = sse_config.instantiate().to(device)
    sse_model.eval()
    sse_batch = [sse_dataset[i] for i in range(8)]
    sse_batch = sse_dataset.collate(sse_batch).to(device)
    sse_losses, sse_states = sse_model(sse_batch, return_states=True)
    
    mse_config = MaskedSpanExtractorConfig(decoder='masked_span_rel', 
                                           bert_like=BertLikeConfig(tokenizer=tokenizer, bert_like=bert, freeze=False, from_subtokenized=True, output_hidden_states=True), 
                                           masked_span_bert_like=MaskedSpanBertLikeConfig(bert_like=bert))
    for entry in conll2004_demo:
        entry['chunks_pred'] = entry['chunks']
    mse_dataset = Dataset(conll2004_demo, mse_config)
    mse_dataset.build_vocabs_and_dims()
    mse_model = mse_config.instantiate().to(device)
    mse_model.eval()
    mse_batch = [mse_dataset[i] for i in range(8)]
    mse_batch = mse_dataset.collate(mse_batch).to(device)
    mse_losses, mse_states = mse_model(mse_batch, return_states=True)
    
    for k, entry in enumerate(conll2004_demo[:8]):
        for i, (label, start, end) in enumerate(entry['chunks']): 
            # Exception for span size 1 if `min_span_size` is 2
            if (min_span_size == 1 or end-start > 1)  and end-start <= 5: 
                delta_hidden = mse_states['span_query_hidden'][k, i] - sse_states['all_query_hidden'][end-start][k, start]
                assert delta_hidden.abs().max().item() < 1e-5
