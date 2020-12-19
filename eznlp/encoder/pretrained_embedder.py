# -*- coding: utf-8 -*-
from typing import Union, List
import torch
from torch.nn.utils.rnn import pad_sequence
import allennlp.modules
import transformers
import flair

from ..data import Batch
from ..nn import SequenceGroupAggregating
from ..nn.functional import seq_lens2mask
from ..config import Config


class PreTrainedEmbedderConfig(Config):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop('arch')
        self.out_dim = kwargs.pop('out_dim')
        self.freeze = kwargs.pop('freeze', True)
        
        if self.arch.lower() == 'elmo':
            self.lstm_stateful = kwargs.pop('lstm_stateful', False)
            self.mix_layers = kwargs.pop('mix_layers', 'trainable')
            self.use_gamma = kwargs.pop('use_gamma', True)
            
        elif self.arch.lower() in ('bert', 'roberta', 'albert'):
            self.tokenizer = kwargs.pop('tokenizer')
            self.agg_mode = kwargs.pop('agg_mode', 'mean')
            self.mix_layers = kwargs.pop('mix_layers', 'top')
            self.use_gamma = kwargs.pop('use_gamma', False)
            
        elif self.arch.lower() == 'flair':
            # start_marker / end_marker in `flair`
            self.sos = kwargs.pop('sos', "\n")
            self.eos = kwargs.pop('eos', " ")
            self.sep = kwargs.pop('sep', " ")
            self.pad = kwargs.pop('pad', " ")
            
            self.agg_mode = kwargs.pop('agg_mode', 'last')
            self.mix_layers = kwargs.pop('mix_layers', 'top')
            self.use_gamma = kwargs.pop('use_gamma', False)
            
        else:
            raise ValueError(f"Invalid pretrained embedder architecture {self.arch}")
        super().__init__(**kwargs)
        
    def instantiate(self, pretrained_model: torch.nn.Module):
        if self.arch.lower() == 'elmo':
            return ELMoEmbedder(self, pretrained_model)
        elif self.arch.lower() in ('bert', 'roberta', 'albert'):
            return BertLikeEmbedder(self, pretrained_model)
        elif self.arch.lower() == 'flair':
            return FlairEmbedder(self, pretrained_model)
        
        
class PreTrainedEmbedder(torch.nn.Module):
    """
    `PreTrainedEmbedder` forwards from inputs to hidden states. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the `pretrained_model` have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    """
    def __init__(self, config: PreTrainedEmbedderConfig, pretrained_model: torch.nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.freeze = config.freeze
        self.mix_layers = config.mix_layers
        self.use_gamma = config.use_gamma
        
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        self.pretrained_model.requires_grad_(not self._freeze)
        
        
        
class ScalarMix(torch.nn.Module):
    """
    Mix multi-layer hidden states by corresponding scalar weights. 
    
    Computes a parameterised scalar mixture of N tensors, 
    ``mixture = gamma * \sum_k(s_k * tensor_k)``
    where ``s = softmax(w)``, with `w` and `gamma` scalar parameters.
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py
    """
    def __init__(self, mix_dim: int):
        super().__init__()
        self.scalars = torch.nn.Parameter(torch.zeros(mix_dim))
        
    def __repr__(self):
        return f"{self.__class__.__name__}(mix_dim={self.scalars.size(0)})"
        
    def forward(self, tensors: Union[torch.FloatTensor, List[torch.FloatTensor]]):
        if isinstance(tensors, (list, tuple)):
            tensors = torch.stack(tensors)
        
        norm_weights_shape = tuple([-1] + [1] * (tensors.dim()-1))
        norm_weights = torch.nn.functional.softmax(self.scalars, dim=0).view(*norm_weights_shape)
        return (tensors * norm_weights).sum(dim=0)
        
        
        
class BertLikeEmbedder(PreTrainedEmbedder):
    """
    An embedder based on BERT representations. 
    
    """
    def __init__(self, config: PreTrainedEmbedderConfig, bert_like: transformers.PreTrainedModel):
        super().__init__(config, bert_like)
        
        self.tokenizer = config.tokenizer
        self.group_aggregating = SequenceGroupAggregating(mode=config.agg_mode)
        if self.mix_layers.lower() == 'trainable':
            self.scalar_mix = ScalarMix(bert_like.config.num_hidden_layers + 1)
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.tensor(1.0))
            
            
    def _build_sub_token_ids(self, tokenized_raw_text: List[str]):
        nested_sub_tokens = [self.tokenizer.tokenize(word) for word in tokenized_raw_text]
        sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        
        sub_tok_ids = [self.tokenizer.cls_token_id] + \
                       self.tokenizer.convert_tokens_to_ids(sub_tokens) + \
                      [self.tokenizer.sep_token_id]
        
        # (step+2, ), (step, )
        return torch.tensor(sub_tok_ids), torch.tensor(ori_indexes)
        
    
    def forward(self, batch: Batch):
        batch_sub_token_data = [self._build_sub_token_ids(text) for text in batch.tokenized_raw_text]
        batch_sub_tok_ids, batch_ori_indexes = list(zip(*batch_sub_token_data))
        sub_tok_seq_lens = torch.tensor([sub_tok_ids.size(0) for sub_tok_ids in batch_sub_tok_ids])
        batch_sub_tok_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = pad_sequence(batch_sub_tok_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_ori_indexes = pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
        
        batch_sub_tok_ids = batch_sub_tok_ids.to(batch.device)
        batch_sub_tok_mask = batch_sub_tok_mask.to(batch.device)
        batch_ori_indexes = batch_ori_indexes.to(batch.device)
        
        # bert_outs: (batch, sub_tok_step+2, hid_dim)
        # hidden: list of (batch, sub_tok_step+2, hid_dim)
        bert_outs, _, hidden = self.pretrained_model(input_ids=batch_sub_tok_ids, 
                                                     attention_mask=(~batch_sub_tok_mask).type(torch.long), 
                                                     output_hidden_states=True)
        
        if self.mix_layers.lower() == 'trainable':
            bert_outs = self.scalar_mix(hidden)
        elif self.mix_layers.lower() == 'top':
            pass
        elif self.mix_layers.lower() == 'average':
            bert_outs = sum(hidden) / len(hidden)
        
        # Remove the `[CLS]` and `[SEP]` positions. 
        bert_outs = bert_outs[:, 1:-1]
        
        # agg_bert_outs: (batch, tok_step, hid_dim)
        agg_bert_outs = self.group_aggregating(bert_outs, batch_ori_indexes, agg_step=batch.tok_ids.size(1))
        if self.use_gamma:
            return self.gamma * agg_bert_outs
        else:
            return agg_bert_outs
    
    
class ELMoEmbedder(PreTrainedEmbedder):
    """
    An embedder based on ELMo representations. 
    
    `Elmo` consists two parts: 
        (1) `_elmo_lstm` includes the (character-based) embedder and (two-layer) BiLSTMs
        (2) `scalar_mix_{i}` includes the "layer weights", which define how different 
            ELMo layers are combined, i.e., `s^{task}` and `gamma^{task}` in Eq. (1) in 
            Peters et al. (2018). This part should always be trainable. 
            
    Setting the `stateful` attribute to False can make the ELMo outputs consistent. 
    
    References
    ----------
    [1] Peters et al. 2018. Deep contextualized word representations. 
    [2] https://github.com/allenai/allennlp/issues/2398
    """
    def __init__(self, config: PreTrainedEmbedderConfig, elmo: allennlp.modules.elmo.Elmo):
        elmo._elmo_lstm._elmo_lstm.stateful = config.lstm_stateful
        
        if config.mix_layers.lower() != 'trainable':
            elmo.scalar_mix_0.scalar_parameters.requires_grad_(False)
            if config.mix_layers.lower() == 'top':
                for scalar_param in elmo.scalar_mix_0.scalar_parameters[:-1]:
                    scalar_param.data.fill_(-9e10)
            
        if not config.use_gamma:
            elmo.scalar_mix_0.gamma.requires_grad_(False)
        
        super().__init__(config, elmo)
        
        
    @property
    def freeze(self):
        return self._freeze
    
    @freeze.setter
    def freeze(self, value: bool):
        assert isinstance(value, bool)
        self._freeze = value
        self.pretrained_model._elmo_lstm.requires_grad_(not self._freeze)
        
    def forward(self, batch: Batch):
        # TODO: use `word_inputs`?
        elmo_char_ids = allennlp.modules.elmo.batch_to_ids(batch.tokenized_raw_text)
        elmo_char_ids = elmo_char_ids.to(batch.device)
        elmo_outs = self.pretrained_model(inputs=elmo_char_ids)
        
        return elmo_outs['elmo_representations'][0]
    
    
    
class FlairEmbedder(PreTrainedEmbedder):
    """
    An embedder based on flair representations. 
    
    References
    ----------
    [1] Akbik et al. 2018. Contextual string embeddings for sequence labeling. 
    [2] https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py
    [3] https://github.com/flairNLP/flair/blob/master/flair/models/language_model.py
    """
    def __init__(self, config: PreTrainedEmbedderConfig, flair_lm: flair.models.LanguageModel):
        super().__init__(config, flair_lm)
        self.sos = config.sos
        self.eos = config.eos
        self.sep = config.sep
        self.pad = config.pad
        self.pad_id = flair_lm.dictionary.get_idx_for_item(self.pad)
        
        self.is_forward = flair_lm.is_forward_lm
        self.dictionary = flair_lm.dictionary
        self.group_aggregating = SequenceGroupAggregating(mode=config.agg_mode)
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.tensor(1.0))
        
        
    def _build_char_ids(self, tokenized_raw_text: List[str]):
        if not self.is_forward:
            tokenized_raw_text = [tok[::-1] for tok in tokenized_raw_text[::-1]]
            
        padded_text = self.sos + self.sep.join(tokenized_raw_text) + self.eos
        ori_indexes = [i for i, tok in enumerate(tokenized_raw_text) for _ in range(len(tok)+len(self.sep))]
        if not self.is_forward:
            ori_indexes = [max(ori_indexes) - i for i in ori_indexes]
        
        ori_indexes = [-1] * len(self.sos) + ori_indexes + [-1] * (len(self.eos) - 1)
        
        char_ids = self.dictionary.get_idx_for_items(padded_text)
        # (char_step, ), (char_step, )
        return torch.tensor(char_ids), torch.tensor(ori_indexes)
        
    
    def forward(self, batch: Batch):
        batch_char_data = [self._build_char_ids(text) for text in batch.tokenized_raw_text]
        batch_char_ids, batch_ori_indexes = list(zip(*batch_char_data))
        
        # char_seq_lens = torch.tensor([char_ids.size(0) for char_ids in batch_char_ids])
        batch_char_ids = pad_sequence(batch_char_ids, batch_first=False, padding_value=self.pad_id)
        batch_ori_indexes = pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
        
        batch_char_ids = batch_char_ids.to(batch.device)
        batch_ori_indexes = batch_ori_indexes.to(batch.device)
        
        # flair_hidden: (char_step, batch, hid_dim)
        _, flair_hidden, _ = self.pretrained_model(batch_char_ids, hidden=None)
        # agg_flair_hidden: (batch, tok_step, hid_dim)
        agg_flair_hidden = self.group_aggregating(flair_hidden.permute(1, 0, 2), batch_ori_indexes, agg_step=batch.tok_ids.size(1))
        
        if self.use_gamma:
            return self.gamma * agg_flair_hidden
        else:
            return agg_flair_hidden
        
        
        