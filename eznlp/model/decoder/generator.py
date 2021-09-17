# -*- coding: utf-8 -*-
from typing import List
import itertools
import nltk
import torch
import torchtext

from ...wrapper import Batch
from ...nn.functional import sequence_pooling
from ...nn.modules import CombinedDropout, SequenceAttention
from ...nn.init import reinit_layer_, reinit_lstm_, reinit_gru_
from ..embedder import OneHotConfig
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase


class GeneratorMixin(DecoderMixinBase):
    @property
    def vocab(self):
        if isinstance(self, GeneratorConfig):
            return self.embedding.vocab
        else:
            return self._vocab
        
    @vocab.setter
    def vocab(self, vocab: torchtext.vocab.Vocab):
        if isinstance(self, GeneratorConfig):
            self.embedding.vocab = vocab
        else:
            self._vocab = vocab
        
        
    def exemplify(self, entry: dict, training: bool=True):
        example = {}
        
        if training:
            # Single ground-truth sentence for training
            example['trg_tok_ids'] = self.embedding.exemplify(entry['trg_tokens'])
        else:
            assert 'trg_tokens' not in entry
            # Notes: The padding positions are ignored in loss computation, so the dev. loss will always be 0
            example['trg_tok_ids'] = torch.tensor([self.embedding.sos_idx] + [self.embedding.pad_idx]*(self.max_len+1), dtype=torch.long)
        
        if 'full_trg_tokens' in entry:
            # Multiple reference sentences for evaluation
            example['full_trg_tokenized_text'] = [tokens.text for tokens in entry['full_trg_tokens']]
        
        return example
        
        
    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch['trg_tok_ids'] = self.embedding.batchify([ex['trg_tok_ids'] for ex in batch_examples])
        
        if 'full_trg_tokenized_text' in batch_examples[0]:
            batch['full_trg_tokenized_text'] = [ex['full_trg_tokenized_text'] for ex in batch_examples]
        
        return batch
        
        
    def retrieve(self, batch: Batch):
        return batch.full_trg_tokenized_text
        
    def evaluate(self, y_gold: List[List[List[str]]], y_pred: List[List[str]]):
        assert isinstance(y_gold[0], list) and isinstance(y_gold[0][0], list) and isinstance(y_gold[0][0][0], str)
        assert isinstance(y_pred[0], list) and isinstance(y_pred[0][0], str)
        # return torchtext.data.metrics.bleu_score(candidate_corpus=y_pred, references_corpus=y_gold)
        return nltk.translate.bleu_score.corpus_bleu(list_of_references=y_gold, hypotheses=y_pred)



class GeneratorConfig(SingleDecoderConfigBase, GeneratorMixin):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch', 'LSTM')
        
        self.embedding = kwargs.pop('embedding', OneHotConfig(tokens_key='trg_tokens', field='text', has_sos=True, has_eos=True))
        # Attention is the default structure
        self.ctx_dim = kwargs.pop('ctx_dim', None)
        self.scoring = kwargs.pop('scoring', 'biaffine')
        
        self.hid_dim = kwargs.pop('hid_dim', 128)
        if self.arch.lower() in ('lstm', 'gru'):
            self.num_layers = kwargs.pop('num_layers', 1)
            self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop('hid_drop_rate', 0.5)
        
        self.shortcut = kwargs.pop('shortcut', False)
        self.teacher_forcing_rate = kwargs.pop('teacher_forcing_rate', 0.5)
        self.max_len = kwargs.pop('max_len', None)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self._name_sep.join([self.scoring, self.arch, self.criterion])
        
    @property
    def in_dim(self):
        return self.ctx_dim
        
    @in_dim.setter
    def in_dim(self, dim: int):
        self.ctx_dim = dim
        
    @property
    def emb_dim(self):
        return self.embedding.emb_dim
        
    @property
    def full_hid_dim(self):
        if not self.shortcut:
            return self.hid_dim
        else:
            return self.hid_dim + self.ctx_dim + self.emb_dim
        
    def build_vocab(self, *partitions):
        flattened = [[{'trg_tokens': tokens} for entry in data for tokens in entry['full_trg_tokens']] for data in partitions]
        self.embedding.build_vocab(*flattened)
        self.max_len = max(len(entry['trg_tokens']) for data in flattened for entry in data)
        
    def instantiate(self):
        if self.arch.lower() in ('lstm', 'gru'):
            return RNNGenerator(self)



class Generator(DecoderBase, GeneratorMixin):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.vocab = config.vocab
        self.teacher_forcing_rate = config.teacher_forcing_rate
        self.shortcut = config.shortcut
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.embedding = config.embedding.instantiate()
        self.attention = SequenceAttention(key_dim=config.ctx_dim, query_dim=config.hid_dim, scoring=config.scoring, external_query=True)
        self.hid2logit = torch.nn.Linear(config.full_hid_dim, config.embedding.voc_dim)
        self.criterion = config.instantiate_criterion(ignore_index=config.embedding.pad_idx, reduction='mean')
        
        
    def forward2logits(self, batch: Batch, src_hidden: torch.Tensor, src_mask: torch.Tensor=None):
        raise NotImplementedError("Not Implemented `forward2logits`")
        
        
    def forward(self, batch: Batch, src_hidden: torch.Tensor=None, src_mask: torch.Tensor=None, logits: torch.Tensor=None):
        # trg_tok_ids: (batch, trg_step)
        # logits: (batch, trg_step-1, voc_dim)
        # atten_weights: (batch, trg_step-1, src_step)
        if logits is None:
            assert src_hidden is not None
            logits = self.forward2logits(batch, src_hidden, src_mask)
        
        # `logits` accords with steps from 1 to `trg_step`
        # Positions of `pad_idx` will be ignored by `self.criterion`
        losses = [self.criterion(lg, t) for lg, t in zip(logits, batch.trg_tok_ids[:, 1:])]
        return torch.stack(losses, dim=0)
        
        
    def decode(self, batch: Batch, src_hidden: torch.Tensor=None, src_mask: torch.Tensor=None, logits: torch.Tensor=None):
        if logits is None:
            assert src_hidden is not None
            logits = self.forward2logits(batch, src_hidden, src_mask)
        
        batch_tok_ids = logits.argmax(dim=-1)
        batch_toks = []
        for tok_ids in batch_tok_ids.cpu().tolist():
            toks = [self.vocab.itos[tok_id] for tok_id in tok_ids]
            toks = list(itertools.takewhile(lambda tok: tok!='<eos>', toks))
            batch_toks.append(toks)
        return batch_toks



class RNNGenerator(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        
        rnn_config = {'input_size': config.ctx_dim + config.emb_dim, 
                      'hidden_size': config.hid_dim, 
                      'num_layers': config.num_layers, 
                      'batch_first': True, 
                      'bidirectional': False, 
                      'dropout': 0.0 if config.num_layers <= 1 else config.hid_drop_rate}
        
        if config.arch.lower() == 'lstm':
            self.rnn = torch.nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config.arch.lower() == 'gru':
            self.rnn = torch.nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)
        
        self.ctx2h_0 = torch.nn.Linear(config.ctx_dim, config.hid_dim*config.num_layers)
        reinit_layer_(self.ctx2h_0, 'tanh')
        if isinstance(self.rnn, torch.nn.LSTM):
            self.ctx2c_0 = torch.nn.Linear(config.ctx_dim, config.hid_dim*config.num_layers)
            reinit_layer_(self.ctx2c_0, 'tanh')
        # RNN hidden states are tanh-activated
        self.tanh = torch.nn.Tanh()
        
        
    def _get_top_hidden(self, h_t: torch.Tensor):
        if isinstance(self.rnn, torch.nn.LSTM):
            # h_t: (h_t, c_t)
            return h_t[0][-1].unsqueeze(1)
        else:
            # h_t: (num_layers, batch, hid_dim) -> (batch, step=1, hid_dim)
            return h_t[-1].unsqueeze(1)
        
    def _init_hidden(self, src_hidden: torch.Tensor, src_mask: torch.Tensor=None):
        context_0 = sequence_pooling(src_hidden, src_mask, mode='mean')
        h_0 = self.tanh(self.ctx2h_0(context_0))
        h_0 = h_0.view(-1, self.rnn.num_layers, self.rnn.hidden_size).permute(1, 0, 2).contiguous()
        if isinstance(self.rnn, torch.nn.LSTM):
            c_0 = self.tanh(self.ctx2c_0(context_0))
            c_0 = c_0.view(-1, self.rnn.num_layers, self.rnn.hidden_size).permute(1, 0, 2).contiguous()
            h_0 = (h_0, c_0)
        return h_0
        
    def forward_step(self, x_t: torch.Tensor, h_tm1: torch.Tensor, src_hidden: torch.Tensor, src_mask: torch.Tensor=None):
        # x_t: (batch, step=1)
        # h_tm1/h_t: (num_layers, batch, hid_dim)
        # src_hidden: (batch, src_step, ctx_dim)
        # embedded_t: (batch, step=1, emb_dim)
        embedded_t = self.dropout(self.embedding(x_t))
        
        # query_t: (batch, step=1, hid_dim)
        # context_t: (batch, step=1, ctx_dim)
        context_t, atten_weight_t = self.attention(src_hidden, mask=src_mask, query=self._get_top_hidden(h_tm1), return_atten_weight=True)
        
        # Forward one RNN step
        _, h_t = self.rnn(torch.cat([embedded_t, context_t], dim=-1), h_tm1)
        
        hidden_t = self._get_top_hidden(h_t)
        if self.shortcut:
            hidden_t = torch.cat([hidden_t, embedded_t, context_t], dim=-1)
        
        # logits_t: (batch, step=1, voc_dim)
        logits_t = self.hid2logit(hidden_t)
        return logits_t, h_t, atten_weight_t
        
        
    def forward2logits(self, batch: Batch, src_hidden: torch.Tensor, src_mask: torch.Tensor=None, return_atten_weight: bool=False):
        # trg_tok_ids: (batch, trg_step)
        # src_hidden: (batch, src_step, ctx_dim)
        h_0 = self._init_hidden(src_hidden, src_mask)
        
        logits, atten_weights = [], []
        # Prepare for step 1
        x_t = batch.trg_tok_ids[:, 0].unsqueeze(1)
        h_tm1 = h_0
        
        for t in range(1, batch.trg_tok_ids.size(1)):
            # t: 1, 2, ..., T-1
            logits_t, h_t, atten_weight_t = self.forward_step(x_t, h_tm1, src_hidden, src_mask)
        
            logits.append(logits_t)
            atten_weights.append(atten_weight_t)
            top1 = logits_t.argmax(dim=-1)
        
            # Prepare for step t+1
            if self.training:
                tf_mask = torch.empty_like(top1).bernoulli(p=self.teacher_forcing_rate).type(torch.bool)
                x_t = torch.where(tf_mask, batch.trg_tok_ids[:, t].unsqueeze(1), top1)
            else:
                x_t = top1
            
            h_tm1 = h_t
        
        # logits: (batch, trg_step-1, voc_dim)
        # atten_weights: (batch, trg_step-1, src_step)
        logits = torch.cat(logits, dim=1)
        atten_weights = torch.cat(atten_weights, dim=1)
        if return_atten_weight:
            return logits, atten_weights
        else:
            return logits
