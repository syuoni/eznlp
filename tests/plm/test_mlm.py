# -*- coding: utf-8 -*-
import pytest
import jieba
import torch
import transformers

from eznlp.io import RawTextIO
from eznlp.plm import MaskedLMConfig
from eznlp.dataset import PreTrainingDataset
from eznlp.training import MaskedLMTrainer


class TestMaskedLM(object):
    def _assert_batch_consistency(self):
        self.model.eval()
        
        batch_inputs012 = {'input_ids': self.batch.mlm_tok_ids[:3], 
                           'attention_mask': (~self.batch.mlm_att_mask[:3]).long(), 
                           'labels': self.batch.mlm_lab_ids[:3]}
        batch_inputs123 = {'input_ids': self.batch.mlm_tok_ids[1:], 
                           'attention_mask': (~self.batch.mlm_att_mask[1:]).long(), 
                           'labels': self.batch.mlm_lab_ids[1:]}
        
        if hasattr(self.batch, 'paired_lab_ids'):
            batch_inputs012.update({'token_type_ids': self.batch.tok_type_ids[:3], 
                                    'next_sentence_label': self.batch.paired_lab_ids[:3]})
            batch_inputs123.update({'token_type_ids': self.batch.tok_type_ids[1:], 
                                    'next_sentence_label': self.batch.paired_lab_ids[1:]})
        
        bert_like_outs012 = self.model(**batch_inputs012)
        bert_like_outs123 = self.model(**batch_inputs123)
        
        if hasattr(self.batch, 'paired_lab_ids'):
            mlm_logits012 = bert_like_outs012['prediction_logits']
            mlm_logits123 = bert_like_outs123['prediction_logits']
        else:
            mlm_logits012 = bert_like_outs012['logits']
            mlm_logits123 = bert_like_outs123['logits']
        
        min_step = min(mlm_logits012.size(1), mlm_logits123.size(1))
        delta_mlm_logits = mlm_logits012[1:, :min_step] - mlm_logits123[:-1, :min_step]
        assert delta_mlm_logits.abs().max().item() < 1e-4
        
    def _assert_trainable(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        trainer = MaskedLMTrainer(self.model, optimizer=optimizer, device=self.device)
        trainer.train_epoch([self.batch])
        
        
    def test_ResumeNER(self, ResumeNER_demo, device):
        PATH = "assets/transformers/bert-base-chinese"
        bert_like = transformers.BertForMaskedLM.from_pretrained(PATH)
        tokenizer = transformers.BertTokenizer.from_pretrained(PATH)
        self.config = MaskedLMConfig(bert_like=bert_like, tokenizer=tokenizer)
        
        self.device = device
        io = RawTextIO(tokenizer.tokenize, jieba.tokenize, max_len=128, document_sep_starts=["-DOCSTART-", "<doc", "</doc"], encoding='utf-8')
        data = io.setup_data_with_tokens(ResumeNER_demo)
        self.dataset = PreTrainingDataset(data, self.config)
        self.model = self.config.instantiate().to(self.device)
        
        self.batch = self.dataset.collate([self.dataset[i] for i in range(4)]).to(self.device)
        self._assert_batch_consistency()
        self._assert_trainable()
        
        
    @pytest.mark.parametrize("use_wwm", [False, True])
    @pytest.mark.parametrize("ngram_weights", [(1.0, ), (0.4, 0.3, 0.3)])
    @pytest.mark.parametrize("paired_task", ['None', 'NSP', 'SOP'])
    def test_wikipedia(self, use_wwm, ngram_weights, paired_task, device):
        PATH = "assets/transformers/bert-base-chinese"
        if paired_task.lower() == 'none':
            bert_like = transformers.BertForMaskedLM.from_pretrained(PATH)
        else:
            bert_like = transformers.BertForPreTraining.from_pretrained(PATH)
        tokenizer = transformers.BertTokenizer.from_pretrained(PATH)
        self.config = MaskedLMConfig(bert_like=bert_like, tokenizer=tokenizer, use_wwm=use_wwm, ngram_weights=ngram_weights, paired_task=paired_task)
        
        self.device = device
        io = RawTextIO(tokenizer.tokenize, jieba.tokenize, max_len=128, document_sep_starts=["-DOCSTART-", "<doc", "</doc"], encoding='utf-8')
        data = io.read("data/Wikipedia/text-zh/AA/wiki_00")
        self.dataset = PreTrainingDataset(data, self.config)
        self.model = self.config.instantiate().to(self.device)
        
        self.batch = self.dataset.collate([self.dataset[i] for i in range(4)]).to(self.device)
        self._assert_batch_consistency()
        self._assert_trainable()
