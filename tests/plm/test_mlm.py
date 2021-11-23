# -*- coding: utf-8 -*-
import torch

from eznlp.io import RawTextIO
from eznlp.plm import MaskedLMConfig
from eznlp.dataset import PreTrainingDataset
from eznlp.training import MaskedLMTrainer


class TestMaskedLM(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        bert_like4mlm_outs012 = self.model(input_ids=self.batch.mlm_tok_ids[:3], 
                                           attention_mask=(~self.batch.mlm_att_mask[:3]).long(), 
                                           labels=self.batch.mlm_lab_ids[:3])
        bert_like4mlm_outs123 = self.model(input_ids=self.batch.mlm_tok_ids[1:], 
                                           attention_mask=(~self.batch.mlm_att_mask[1:]).long(), 
                                           labels=self.batch.mlm_lab_ids[1:])
        mlm_logits012 = bert_like4mlm_outs012['logits']
        mlm_logits123 = bert_like4mlm_outs123['logits']
        
        min_step = min(mlm_logits012.size(1), mlm_logits123.size(1))
        delta_mlm_logits = mlm_logits012[1:, :min_step] - mlm_logits123[:-1, :min_step]
        assert delta_mlm_logits.abs().max().item() < 1e-4
        
    def _assert_trainable(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        trainer = MaskedLMTrainer(self.model, optimizer=optimizer, device=self.device)
        trainer.train_epoch([self.batch])
        
        
    def test_conll2003(self, bert_like4mlm_with_tokenizer, conll2003_demo, device):
        bert_like4mlm, tokenizer = bert_like4mlm_with_tokenizer
        self.config = MaskedLMConfig(bert_like=bert_like4mlm, tokenizer=tokenizer)
        
        self.device = device
        self.dataset = PreTrainingDataset(conll2003_demo, self.config)
        self.model = self.config.instantiate().to(self.device)
        
        self.batch = self.dataset.collate([self.dataset[i] for i in range(4)]).to(self.device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    def test_PMC(self, bert_like4mlm_with_tokenizer, device):
        bert_like4mlm, tokenizer = bert_like4mlm_with_tokenizer
        self.config = MaskedLMConfig(bert_like=bert_like4mlm, tokenizer=tokenizer)
        
        self.device = device
        io = RawTextIO(tokenizer.tokenize, max_len=128, encoding='utf-8')
        data = io.read("data/PMC/comm_use/Cells/Cells_2012_Apr_23_1(2)_35-50.txt")
        self.dataset = PreTrainingDataset(data, self.config)
        self.model = self.config.instantiate().to(self.device)
        
        self.batch = self.dataset.collate([self.dataset[i] for i in range(4)]).to(self.device)
        self._assert_batch_consistency()
        self._assert_trainable()
