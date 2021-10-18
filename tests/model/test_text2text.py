# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.dataset import GenerationDataset
from eznlp.training import Trainer
from eznlp.model import OneHotConfig, EncoderConfig, GeneratorConfig, Text2TextConfig


class TestModel(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch = [self.dataset[i] for i in range(4)]
        batch012 = self.dataset.collate(batch[:3]).to(self.device)
        batch123 = self.dataset.collate(batch[1:]).to(self.device)
        losses012, states012 = self.model(batch012, return_states=True)
        losses123, states123 = self.model(batch123, return_states=True)
        
        logits012, logits123 = states012['logits'], states123['logits']
        min_step = min(logits012.size(1), logits123.size(1))
        delta_logits = logits012[1:, :min_step] - logits123[:-1, :min_step]
        assert delta_logits.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        pred012 = self.model.decode(batch012, **states012)
        pred123 = self.model.decode(batch123, **states123)
        pred012 = [yi[:min_step] for yi in pred012]
        pred123 = [yi[:min_step] for yi in pred123]
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
        
        self.dataset = GenerationDataset(data, self.config)
        self.dataset.build_vocabs_and_dims()
        self.model = self.config.instantiate().to(self.device)
        assert isinstance(self.config.name, str) and len(self.config.name) > 0
        
        
    @pytest.mark.parametrize("arch", ['LSTM', 'GRU', 'Gehring', 'Transformer'])
    @pytest.mark.parametrize("num_heads", [1, 4])
    @pytest.mark.parametrize("shortcut", [False, True])
    def test_model(self, arch, num_heads, shortcut, multi30k_demo, device):
        self.config = Text2TextConfig(encoder=EncoderConfig(arch=arch, use_emb2init_hid=True), 
                                      decoder=GeneratorConfig(arch=arch, use_emb2init_hid=True, num_heads=num_heads, shortcut=shortcut))
        self._setup_case(multi30k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("init_ctx_mode", ['mean_pooling', 'attention', 'rnn_last'])
    def test_model_with_rnn(self, init_ctx_mode, multi30k_demo, device):
        self.config = Text2TextConfig(decoder=GeneratorConfig(arch='LSTM', init_ctx_mode=init_ctx_mode))
        self._setup_case(multi30k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("use_emb2init_hid", [False, True])
    @pytest.mark.parametrize("sin_positional_emb", [False, True])
    @pytest.mark.parametrize("weight_tying", [False, True])
    def test_model_with_transformer(self, use_emb2init_hid, sin_positional_emb, weight_tying, multi30k_demo, device):
        self.config = Text2TextConfig(embedder=OneHotConfig(tokens_key='tokens', field='text', emb_dim=128, 
                                                            has_positional_emb=True, sin_positional_emb=sin_positional_emb),  
                                      encoder=EncoderConfig(arch='Transformer', use_emb2init_hid=use_emb2init_hid, hid_dim=128), 
                                      decoder=GeneratorConfig(embedding=OneHotConfig(tokens_key='trg_tokens', field='text', emb_dim=128, has_sos=True, has_eos=True, 
                                                                                     has_positional_emb=True, sin_positional_emb=sin_positional_emb), 
                                                              arch='Transformer', use_emb2init_hid=use_emb2init_hid, hid_dim=128, weight_tying=weight_tying))
        self._setup_case(multi30k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
    @pytest.mark.parametrize("sl_epsilon", [0.0, 0.1, 0.2])
    def test_model_with_label_smoothing(self, sl_epsilon, multi30k_demo, device):
        self.config = Text2TextConfig(decoder=GeneratorConfig(arch='LSTM', sl_epsilon=sl_epsilon))
        self._setup_case(multi30k_demo, device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("arch", ['Gehring', 'Transformer'])
    def test_forward_consistency(self, arch, multi30k_demo, device):
        # Note `Generator.forward2logits_step_by_step` always uses predicted token as the input in the eval mode
        # Hence, here manually set all dropout rate to be 0 and use training mode. 
        self.config = Text2TextConfig(encoder=EncoderConfig(arch=arch, use_emb2init_hid=True, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0), 
                                      decoder=GeneratorConfig(arch=arch, use_emb2init_hid=True, num_layers=1, teacher_forcing_rate=1.0, 
                                                              in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0))
        self._setup_case(multi30k_demo, device)
        
        self.model.train()
        batch = [self.dataset[i] for i in range(4)]
        batch = self.dataset.collate(batch).to(self.device)
        
        states = self.model.forward2states(batch)
        del states['logits']
        logits_sbs = self.model.decoder.forward2logits_step_by_step(batch, **states)
        logits_aao = self.model.decoder.forward2logits_all_at_once(batch, **states)
        assert (logits_aao - logits_sbs).abs().max().item() < 1e-5
        
        
    @pytest.mark.parametrize("arch", ['LSTM', 'Gehring', 'Transformer'])
    def test_beam_search(self, arch, multi30k_demo, device):
        self.config = Text2TextConfig(encoder=EncoderConfig(arch=arch, use_emb2init_hid=True), decoder=GeneratorConfig(arch=arch, use_emb2init_hid=True))
        self._setup_case(multi30k_demo, device)
        
        self.model.eval()
        batch = [self.dataset[i] for i in range(4)]
        batch = self.dataset.collate(batch).to(self.device)
        
        greedy_res = self.model.decode(batch)
        beam1_res  = self.model.beam_search(1, batch)
        assert beam1_res == greedy_res
        
        
    @pytest.mark.parametrize("arch", ['LSTM', 'Gehring', 'Transformer'])
    def test_prediction_without_gold(self, arch, multi30k_demo, device):
        self.config = Text2TextConfig(encoder=EncoderConfig(arch=arch, use_emb2init_hid=True), 
                                      decoder=GeneratorConfig(arch=arch, use_emb2init_hid=True))
        self._setup_case(multi30k_demo, device)
        
        data_wo_gold = [{'tokens': entry['tokens']} for entry in multi30k_demo]
        dataset_wo_gold = GenerationDataset(data_wo_gold, self.config, training=False)
        
        trainer = Trainer(self.model, device=device)
        y_pred = trainer.predict(dataset_wo_gold)
        assert len(y_pred) == len(data_wo_gold)
