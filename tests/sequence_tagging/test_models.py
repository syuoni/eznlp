# -*- coding: utf-8 -*-
import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from eznlp import Token
from eznlp import ConfigDict
from eznlp import TokenConfig, CharConfig, EnumConfig, ValConfig, EmbedderConfig
from eznlp import EncoderConfig, PreTrainedEmbedderConfig
from eznlp.sequence_tagging import DecoderConfig, SequenceTaggerConfig
from eznlp.sequence_tagging import SequenceTaggingDataset
from eznlp.sequence_tagging import SequenceTaggingTrainer
from eznlp.sequence_tagging.io import ConllIO


def load_demo_data(scheme='BIOES'):
    conll_io = ConllIO(text_col_id=0, tag_col_id=3, scheme='BIO1')
    train_data = conll_io.read("assets/data/conll2003/eng.train")
    val_data   = conll_io.read("assets/data/conll2003/eng.testa")
    test_data  = conll_io.read("assets/data/conll2003/eng.testb")
    return train_data, val_data, test_data


def build_demo_datasets(train_data, val_data, test_data, config):
    train_set = SequenceTaggingDataset(train_data, config)
    val_set   = SequenceTaggingDataset(val_data,   train_set.config)
    test_set  = SequenceTaggingDataset(test_data,  train_set.config)
    return train_set, val_set, test_set


@pytest.fixture
def BIOES_data():
    return load_demo_data(scheme='BIOES')

@pytest.fixture
def BIO2_data():
    return load_demo_data(scheme='BIO2')



class TestCharEncoder(object):
    @pytest.mark.parametrize("arch", ['CNN', 'LSTM', 'GRU'])
    def test_char_encoder(self, BIOES_data, arch, device):
        train_data, val_data, test_data = BIOES_data
        
        config = SequenceTaggerConfig(embedder=EmbedderConfig(char=CharConfig(arch=arch)))
        assert not config.is_valid
        train_set = SequenceTaggingDataset(train_data, config)
        assert config.is_valid
        
        batch = train_set.collate([train_set[i] for i in range(0, 4)]).to(device)
        tagger = config.instantiate().to(device)
        char_encoder = tagger.embedder.char_encoder
        char_encoder.eval()
        
        batch_seq_lens1 = batch.seq_lens.clone()
        batch_seq_lens1[0] = batch_seq_lens1[0] - 1
        char_feats1 = char_encoder(batch.char_ids[1:], batch.tok_lens[1:], batch.char_mask[1:], batch_seq_lens1)
        
        batch_seq_lens2 = batch.seq_lens.clone()
        batch_seq_lens2[-1] = batch_seq_lens2[-1] - 1
        char_feats2 = char_encoder(batch.char_ids[:-1], batch.tok_lens[:-1], batch.char_mask[:-1], batch_seq_lens2)
        
        step = min(char_feats1.size(1), char_feats2.size(1))
        last_step = batch_seq_lens2[-1].item()
        assert (char_feats1[0, :step-1]  - char_feats2[0, 1:step]).abs().max() < 1e-4
        assert (char_feats1[1:-1, :step] - char_feats2[1:-1, :step]).abs().max() < 1e-4
        assert (char_feats1[-1, :last_step] - char_feats2[-1, :last_step]).abs().max() < 1e-4
        
        
class TestBatching(object):
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA available")
    def test_batch_to_cuda(self, BIOES_data, device):
        train_data, val_data, test_data = BIOES_data
        
        embedder_config = EmbedderConfig(enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token.basic_enum_fields]), 
                                         val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token.basic_val_fields]))
        config = SequenceTaggerConfig(embedder=embedder_config)
        train_set = SequenceTaggingDataset(train_data, config)
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=train_set.collate, pin_memory=True)
        for batch in train_loader:
            break
        
        assert batch.tok_ids.is_pinned()
        assert batch.enum['en_pattern'].is_pinned()
        assert batch.val['en_shape_features'].is_pinned()
        assert batch.seq_lens.is_pinned()
        assert batch.tags_objs[0].tag_ids.is_pinned()
        
        cpu = torch.device('cpu')
        gpu = torch.device('cuda:0')
        assert batch.tok_ids.device == cpu
        batch = batch.to(gpu)
        assert batch.tok_ids.device == gpu
        assert not batch.tok_ids.is_pinned()
        
        
    def test_batches(self, BIOES_data):
        config = SequenceTaggerConfig()
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        train_loader = DataLoader(train_set, batch_size=8, shuffle=True, 
                                  collate_fn=train_set.collate)
        val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, 
                                  collate_fn=val_set.collate)
        test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False, 
                                  collate_fn=test_set.collate)
        for batch in train_loader:
            break
        for batch in val_loader:
            break
        for batch in test_loader:
            break
        
        
class TestTagger(object):
    def one_tagger_pass(self, tagger, train_set, device):
        tagger.eval()
        
        batch012 = train_set.collate([train_set[i] for i in range(0, 3)]).to(device)
        batch123 = train_set.collate([train_set[i] for i in range(1, 4)]).to(device)
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
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_epoch([batch012])
        trainer.eval_epoch([batch012])
        
    
    @pytest.mark.parametrize("freeze", [False, True])
    def test_word_embedding_initialization(self, BIOES_data, glove100, freeze, device):
        embedder_config = EmbedderConfig(token=TokenConfig(emb_dim=100, freeze=freeze))
        config = SequenceTaggerConfig(embedder=embedder_config)
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate(pretrained_vectors=glove100).to(device)
        
        self.one_tagger_pass(tagger, train_set, device)
        
        
    def test_train_steps(self, BIOES_data, device):
        config = SequenceTaggerConfig()
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate().to(device)
        
        batch = train_set.collate([train_set[i] for i in range(0, 4)]).to(device)
        
        optimizer = optim.AdamW(tagger.parameters())
        trainer = SequenceTaggingTrainer(tagger, optimizer=optimizer, device=device)
        trainer.train_steps(train_loader=[batch, batch], 
                            eval_loader=[batch, batch], 
                            n_epochs=10, disp_every_steps=2, eval_every_steps=6)
    
    
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM', 'GRU', 'Transformer'])
    @pytest.mark.parametrize("shortcut", [False, True])
    @pytest.mark.parametrize("dec_arch", ['softmax', 'CRF'])
    def test_tagger(self, BIOES_data, enc_arch, shortcut, dec_arch, device):
        config = SequenceTaggerConfig(encoder=EncoderConfig(arch=enc_arch, shortcut=shortcut), 
                                      decoder=DecoderConfig(arch=dec_arch))
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, train_set, device)
        
        
    def test_tagger_intermediate(self, BIOES_data, device):
        config = SequenceTaggerConfig(intermediate=EncoderConfig())
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, train_set, device)
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_elmo(self, BIOES_data, elmo, freeze, device):
        elmo_embedder_config = PreTrainedEmbedderConfig(arch='ELMo', 
                                                        out_dim=elmo.get_output_dim(), 
                                                        lstm_stateful=False, 
                                                        freeze=freeze)
        config = SequenceTaggerConfig(encoder=None, elmo_embedder=elmo_embedder_config)
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate(elmo=elmo).to(device)
        
        self.one_tagger_pass(tagger, train_set, device)
    
    
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_bert_like(self, BIOES_data, bert_with_tokenizer, freeze, device):
        bert, tokenizer = bert_with_tokenizer
        bert_like_embedder_config = PreTrainedEmbedderConfig(arch='BERT', 
                                                             out_dim=bert.config.hidden_size, 
                                                             tokenizer=tokenizer, 
                                                             freeze=freeze)
        config = SequenceTaggerConfig(encoder=None, bert_like_embedder=bert_like_embedder_config)
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate(bert_like=bert).to(device)
        
        self.one_tagger_pass(tagger, train_set, device)
        
        
    @pytest.mark.parametrize("freeze", [False, True])
    def test_tagger_flair(self, BIOES_data, flair_fw_lm, flair_bw_lm, freeze, device):
        flair_fw_embedder_config = PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_fw_lm.hidden_size, freeze=freeze)
        flair_bw_embedder_config = PreTrainedEmbedderConfig(arch='Flair', out_dim=flair_bw_lm.hidden_size, freeze=freeze)
        
        config = SequenceTaggerConfig(encoder=None, 
                                      flair_fw_embedder=flair_fw_embedder_config, 
                                      flair_bw_embedder=flair_bw_embedder_config)
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate(flair_fw_lm=flair_fw_lm, flair_bw_lm=flair_bw_lm).to(device)
        
        self.one_tagger_pass(tagger, train_set, device)
        
        
    # @pytest.mark.parametrize("dec_arch", ['softmax', 'CRF'])
    # @pytest.mark.parametrize("cascade_mode", ['Sliced', 'Straight'])
    # def test_tagger_cascade(self, BIOES_data, dec_arch, cascade_mode, device):
    #     config = SequenceTaggerConfig(decoder=DecoderConfig(arch=dec_arch, cascade_mode=cascade_mode))
    #     train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
    #     tagger = config.instantiate().to(device)
    #     self.one_tagger_pass(tagger, train_set, device)
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    def test_tagger_morefields(self, BIOES_data, enc_arch, device):
        embedder_config = EmbedderConfig(enum=ConfigDict([(f, EnumConfig(emb_dim=20)) for f in Token.basic_enum_fields]), 
                                         val=ConfigDict([(f, ValConfig(emb_dim=20)) for f in Token.basic_val_fields]))
        config = SequenceTaggerConfig(embedder=embedder_config, 
                                      encoder=EncoderConfig(arch=enc_arch))
        train_set, val_set, test_set = build_demo_datasets(*BIOES_data, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, train_set, device)
        
        
    @pytest.mark.parametrize("enc_arch", ['CNN', 'LSTM'])
    def test_tagger_BIO2(self, BIO2_data, enc_arch, device):
        config = SequenceTaggerConfig(encoder=EncoderConfig(arch=enc_arch))
        train_set, val_set, test_set = build_demo_datasets(*BIO2_data, config)
        tagger = config.instantiate().to(device)
        self.one_tagger_pass(tagger, train_set, device)
        