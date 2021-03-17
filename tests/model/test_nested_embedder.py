# -*- coding: utf-8 -*-
import pytest
import torch

from eznlp.token import LexiconTokenizer
from eznlp.model import EncoderConfig, NestedOneHotConfig, CharConfig, SoftLexiconConfig


class TestNestedOneHotEmbedder(object):
    def _assert_batch_consistency(self):
        self.embedder.eval()
        
        seq_lens1 = self.seq_lens.clone()
        seq_lens1[0] = seq_lens1[0] - 1
        batch1 = {k: v[self.config.num_channels:] for k, v in self.batch.items()}
        embedded1 = self.embedder(**batch1, seq_lens=seq_lens1)
        
        seq_lens2 = self.seq_lens.clone()
        seq_lens2[-1] = seq_lens2[-1] - 1
        batch2 = {k: v[:-self.config.num_channels] for k, v in self.batch.items()}
        embedded2 = self.embedder(**batch2, seq_lens=seq_lens2)
        
        step = min(embedded1.size(1), embedded2.size(1))
        last_step = seq_lens2[-1].item()
        assert (embedded1[0, :step-1]  - embedded2[0, 1:step]).abs().max() < 1e-4
        assert (embedded1[1:-1, :step] - embedded2[1:-1, :step]).abs().max() < 1e-4
        assert (embedded1[-1, :last_step] - embedded2[-1, :last_step]).abs().max() < 1e-4
        
        
    @pytest.mark.parametrize("arch", ['Conv', 'LSTM', 'GRU'])
    def test_char_encoder(self, arch, conll2003_demo):
        self.config = CharConfig(encoder=EncoderConfig(arch=arch, 
                                                       hid_dim=128, 
                                                       num_layers=1, 
                                                       in_drop_rates=(0.5, 0.0, 0.0)))
        self.config.build_vocab(conll2003_demo)
        assert self.config.valid
        
        self.seq_lens = torch.tensor([len(conll2003_demo[i]['tokens']) for i in range(4)])
        batch_ex = [self.config.exemplify(conll2003_demo[i]['tokens']) for i in range(4)]
        self.batch = self.config.batchify(batch_ex)
        
        self.embedder = self.config.instantiate()
        self._assert_batch_consistency()
        
        
    def test_softlexicon(self, ctb50, ResumeNER_demo):
        tokenizer = LexiconTokenizer(ctb50.itos)
        for data_entry in ResumeNER_demo:
            data_entry['tokens'].build_softlexicons(tokenizer.tokenize)
        
        self.config = SoftLexiconConfig(vectors=ctb50)
        self.config.build_vocab(ResumeNER_demo)
        self.config.build_freqs(ResumeNER_demo)
        assert self.config.valid
        
        self.seq_lens = torch.tensor([len(ResumeNER_demo[i]['tokens']) for i in range(4)])
        batch_ex = [self.config.exemplify(ResumeNER_demo[i]['tokens']) for i in range(4)]
        self.batch = self.config.batchify(batch_ex)
        
        self.embedder = self.config.instantiate()
        self._assert_batch_consistency()
        
        