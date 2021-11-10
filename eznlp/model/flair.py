# -*- coding: utf-8 -*-
from typing import List
import torch
import flair

from ..token import TokenSequence
from ..nn.modules import SequenceGroupAggregating
from ..config import Config


class FlairConfig(Config):
    def __init__(self, **kwargs):
        self.flair_lm: flair.models.LanguageModel = kwargs.pop('flair_lm')
        self.out_dim = self.flair_lm.hidden_size
        self.is_forward = self.flair_lm.is_forward_lm
        
        # start_marker / end_marker in `flair`
        self.sos = kwargs.pop('sos', "\n")
        self.eos = kwargs.pop('eos', " ")
        self.sep = kwargs.pop('sep', " ")
        self.pad = kwargs.pop('pad', " ")
        self.dictionary = self.flair_lm.dictionary
        self.pad_id = self.dictionary.get_idx_for_item(self.pad)
        
        self.arch = kwargs.pop('arch', 'Flair')
        self.freeze = kwargs.pop('freeze', True)
        
        self.agg_mode = kwargs.pop('agg_mode', 'last')
        self.use_gamma = kwargs.pop('use_gamma', False)
        
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.arch
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['flair_lm'] = None
        return state
        
        
    def exemplify(self, tokens: TokenSequence):
        tokenized_raw_text = tokens.raw_text
        if not self.is_forward:
            tokenized_raw_text = [tok[::-1] for tok in tokenized_raw_text[::-1]]
        
        padded_text = self.sos + self.sep.join(tokenized_raw_text) + self.eos
        ori_indexes = [i for i, tok in enumerate(tokenized_raw_text) for _ in range(len(tok)+len(self.sep))]
        if not self.is_forward:
            ori_indexes = [max(ori_indexes) - i for i in ori_indexes]
        
        ori_indexes = [-1] * len(self.sos) + ori_indexes + [-1] * (len(self.eos) - 1)
        
        char_ids = self.dictionary.get_idx_for_items(padded_text)
        # (char_step, ), (char_step, )
        return {'char_ids': torch.tensor(char_ids), 
                'ori_indexes': torch.tensor(ori_indexes)}
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_char_ids = [ex['char_ids'] for ex in batch_ex]
        batch_ori_indexes = [ex['ori_indexes'] for ex in batch_ex]
        
        # char_seq_lens = torch.tensor([char_ids.size(0) for char_ids in batch_char_ids])
        batch_char_ids = torch.nn.utils.rnn.pad_sequence(batch_char_ids, batch_first=False, padding_value=self.pad_id)
        batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
        return {'char_ids': batch_char_ids, 
                'ori_indexes': batch_ori_indexes}
        
        
    def instantiate(self):
        return FlairEmbedder(self)



class FlairEmbedder(torch.nn.Module):
    """
    An embedder based on flair representations. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the `pretrained_model` have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    
    References
    ----------
    [1] Akbik et al. 2018. Contextual string embeddings for sequence labeling. 
    [2] https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py
    [3] https://github.com/flairNLP/flair/blob/master/flair/models/language_model.py
    """
    def __init__(self, config: FlairConfig):
        super().__init__()
        self.flair_lm = config.flair_lm
        
        self.freeze = config.freeze
        self.use_gamma = config.use_gamma
        
        self.group_aggregating = SequenceGroupAggregating(mode=config.agg_mode)
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        
        
    @property
    def freeze(self):
        return self._freeze
        
    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.flair_lm.requires_grad_(not freeze)
        
    def forward(self, char_ids: torch.LongTensor, ori_indexes: torch.LongTensor):
        # flair_hidden: (char_step, batch, hid_dim)
        _, flair_hidden, _ = self.flair_lm(char_ids, hidden=None)
        # agg_flair_hidden: (batch, tok_step, hid_dim)
        agg_flair_hidden = self.group_aggregating(flair_hidden.permute(1, 0, 2), ori_indexes)
        
        if self.use_gamma:
            return self.gamma * agg_flair_hidden
        else:
            return agg_flair_hidden
