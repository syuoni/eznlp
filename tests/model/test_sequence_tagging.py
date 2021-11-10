# -*- coding: utf-8 -*-
import pytest
import random
import torch

from eznlp.token import Token, LexiconTokenizer
from eznlp.dataset import Dataset
from eznlp.config import ConfigDict
from eznlp.model import OneHotConfig, MultiHotConfig, EncoderConfig
from eznlp.model import CharConfig, SoftLexiconConfig
from eznlp.model import ELMoConfig, BertLikeConfig, FlairConfig
from eznlp.model import SequenceTaggingDecoderConfig, ExtractorConfig
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
        
        
    @pytest.mark.parametrize("arch", ['FFN', 'Conv', 'Gehring', 'LSTM', 'GRU', 'Transformer'])
    @pytest.mark.parametrize("shortcut", [False, True])
    @pytest.mark.parametrize("use_crf, fl_gamma, sl_epsilon", [(True,  0.0, 0.0), 
                                                               (False, 2.0, 0.0),
                                                               (False, 0.0, 0.1)])
    def test_model(self, arch, shortcut, use_crf, fl_gamma, sl_epsilon, conll2003_demo, device):
        self.config = ExtractorConfig(intermediate2=EncoderConfig(arch=arch, shortcut=shortcut, use_emb2init_hid=True), 
                                      decoder=SequenceTaggingDecoderConfig(use_crf=use_crf, fl_gamma=fl_gamma, sl_epsilon=sl_epsilon))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("arch", ['Conv', 'LSTM'])
    def test_model_with_char(self, arch, conll2003_demo, device):
        char_config = CharConfig(encoder=EncoderConfig(arch=arch, 
                                                       hid_dim=128, 
                                                       num_layers=1, 
                                                       in_drop_rates=(0.5, 0.0, 0.0)))
        self.config = ExtractorConfig('sequence_tagging', nested_ohots=ConfigDict({'char': char_config}), )
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_softlexicon(self, ctb50, ResumeNER_demo, device):
        tokenizer = LexiconTokenizer(ctb50.itos)
        for data_entry in ResumeNER_demo:
            data_entry['tokens'].build_softlexicons(tokenizer.tokenize)
        
        self.config = ExtractorConfig('sequence_tagging', nested_ohots=ConfigDict({'softlexicon': SoftLexiconConfig(vectors=ctb50)}))
        self._setup_case(ResumeNER_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_model_with_intermediate(self, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging', 
                                      intermediate1=EncoderConfig(), 
                                      intermediate2=EncoderConfig())
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    def test_model_with_more_fields(self, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging', 
                                      ohots=ConfigDict({f: OneHotConfig(field=f, emb_dim=20) for f in Token._basic_ohot_fields}), 
                                      mhots=ConfigDict({f: MultiHotConfig(field=f, emb_dim=20) for f in Token._basic_mhot_fields}))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("scheme", ['BIO1', 'BIO2'])
    def test_model_with_alternative_schemes(self, scheme, conll2003_demo, device):
        self.config = ExtractorConfig(decoder=SequenceTaggingDecoderConfig(scheme=scheme))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_model_with_pretrained_vector(self, freeze, glove100, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging', ohots=ConfigDict({'text': OneHotConfig(field='text', vectors=glove100, freeze=freeze)}))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_model_with_elmo(self, freeze, elmo, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging', ohots=None, elmo=ELMoConfig(elmo=elmo))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_model_with_bert_like(self, freeze, bert_like_with_tokenizer, conll2003_demo, device):
        bert_like, tokenizer = bert_like_with_tokenizer
        self.config = ExtractorConfig('sequence_tagging', ohots=None, 
                                      bert_like=BertLikeConfig(bert_like=bert_like, tokenizer=tokenizer, freeze=freeze))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_model_with_flair(self, freeze, flair_fw_lm, flair_bw_lm, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging', ohots=None, 
                                      flair_fw=FlairConfig(flair_lm=flair_fw_lm, freeze=freeze),  
                                      flair_bw=FlairConfig(flair_lm=flair_bw_lm, freeze=freeze))
        self._setup_case(conll2003_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_prediction_without_gold(self, conll2003_demo, device):
        self.config = ExtractorConfig('sequence_tagging')
        self._setup_case(conll2003_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in conll2003_demo]
        dataset_wo_gold = Dataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        set_chunks_pred = trainer.predict(dataset_wo_gold)
        assert len(set_chunks_pred) == len(data_wo_gold)
