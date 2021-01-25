# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.data import Token
from eznlp import ConfigDict
from eznlp import TokenConfig, EnumConfig, ValConfig, EmbedderConfig
from eznlp import EncoderConfig, PreTrainedEmbedderConfig
from eznlp.sequence_tagging import SequenceTaggingDecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer



class TestTagger(object):
    def one_tagger_pass(self, tagger, dataset, device):
        tagger.eval()
        
        batch012 = dataset.collate([dataset[i] for i in range(0, 3)]).to(device)
        batch123 = dataset.collate([dataset[i] for i in range(1, 4)]).to(device)
        losses012, hidden012 = tagger(batch012, return_hidden=True)
        losses123, hidden123 = tagger(batch123, return_hidden=True)
        
        min_step = min(hidden012.size(1), hidden123.size(1))
        delta_hidden = hidden012[1:, :min_step] - hidden123[:-1, :min_step]
        assert delta_hidden.abs().max().item() < 1e-4
        
        delta_losses = losses012[1:] - losses123[:-1]
        assert delta_losses.abs().max().item() < 2e-4
        
        best_paths012 = tagger.decode(batch012)
        best_paths123 = tagger.decode(batch123)
        assert best_paths012[1:] == best_paths123[:-1]
        
        optimizer = torch.optim.AdamW(tagger.parameters())
        trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM', 'GRU', 'Transformer'])
    @pytest.mark.parametrize("shortcut", [False, True])
    @pytest.mark.parametrize("dec_arch", ['softmax', 'CRF'])
    def test_tagger(self, conll2003_demo, enc_arch, shortcut, dec_arch, device):
        config = SequenceTaggerConfig(encoder=EncoderConfig(arch=enc_arch, shortcut=shortcut), 
                                      decoder=SequenceTaggingDecoderConfig(arch=dec_arch))
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_word_embedding_initialization(self, conll2003_demo, glove100, freeze, device):
        embedder_config = EmbedderConfig(token=TokenConfig(emb_dim=100, freeze=freeze))
        config = SequenceTaggerConfig(embedder=embedder_config)
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate(pretrained_vectors=glove100).to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    def test_tagger_intermediate(self, conll2003_demo, device):
        config = SequenceTaggerConfig(intermediate=EncoderConfig())
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_elmo(self, conll2003_demo, elmo, freeze, device):
        elmo_embedder_config = PreTrainedEmbedderConfig(arch='ELMo', 
                                                        out_dim=elmo.get_output_dim(), 
                                                        lstm_stateful=False, 
                                                        freeze=freeze)
        config = SequenceTaggerConfig(encoder=None, elmo_embedder=elmo_embedder_config)
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate(elmo=elmo).to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
    
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_bert_like(self, conll2003_demo, bert_with_tokenizer, freeze, device):
        bert, tokenizer = bert_with_tokenizer
        bert_like_embedder_config = PreTrainedEmbedderConfig(arch='BERT', 
                                                             out_dim=bert.config.hidden_size, 
                                                             tokenizer=tokenizer, 
                                                             freeze=freeze)
        config = SequenceTaggerConfig(encoder=None, bert_like_embedder=bert_like_embedder_config)
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate(bert_like=bert).to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_flair(self, conll2003_demo, flair_fw_lm, flair_bw_lm, freeze, device):
        flair_fw_embedder_config = PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_fw_lm.hidden_size, freeze=freeze)
        flair_bw_embedder_config = PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_bw_lm.hidden_size, freeze=freeze)
        config = SequenceTaggerConfig(encoder=None, 
                                      flair_fw_embedder=flair_fw_embedder_config, 
                                      flair_bw_embedder=flair_bw_embedder_config)
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate(flair_fw_lm=flair_fw_lm, flair_bw_lm=flair_bw_lm).to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    def test_tagger_morefields(self, conll2003_demo, enc_arch, device):
        embedder_config = EmbedderConfig(enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token._basic_enum_fields]), 
                                         val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token._basic_val_fields]))
        config = SequenceTaggerConfig(embedder=embedder_config, 
                                      encoder=EncoderConfig(arch=enc_arch))
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    @pytest.mark.parametrize("scheme", ['BIO1', 'BIO2'])
    def test_tagger_schemes(self, conll2003_demo, scheme, device):
        config = SequenceTaggerConfig(decoder=SequenceTaggingDecoderConfig(scheme=scheme))
        
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, dataset, device)
        
        
    def test_train_steps(self, conll2003_demo, device):
        dataset = SequenceTaggingDataset(conll2003_demo)
        tagger = dataset.config.instantiate().to(device)
        
        batch = dataset.collate([dataset[i] for i in range(0, 4)]).to(device)
        
        optimizer = torch.optim.AdamW(tagger.parameters())
        trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_steps(train_loader=[batch, batch], eval_loader=[batch, batch], 
                            n_epochs=10, disp_every_steps=2, eval_every_steps=6)
        