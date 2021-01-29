# -*- coding: utf-8 -*-
import pytest

from eznlp import CharConfig, EmbedderConfig
from eznlp.sequence_tagging import SequenceTaggerConfig, SequenceTaggingDataset


class TestCharEncoder(object):
    @pytest.mark.parametrize("arch", ['CNN', 'LSTM', 'GRU'])
    def test_char_encoder(self, conll2003_demo, arch, device):
        config = SequenceTaggerConfig(embedder=EmbedderConfig(char=CharConfig(arch=arch)))
        assert not config.is_valid
        dataset = SequenceTaggingDataset(conll2003_demo, config)
        assert config.is_valid
        
        batch = dataset.collate([dataset[i] for i in range(0, 4)]).to(device)
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
        
        