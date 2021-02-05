# -*- coding: utf-8 -*-
from typing import List
import torch
import transformers

from ..data.token import TokenSequence
from ..nn.modules import SequenceGroupAggregating, ScalarMix
from ..nn.functional import seq_lens2mask
from ..config import Config



class BertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        self.out_dim = self.bert_like.config.hidden_size
        self.num_hidden_layers = self.bert_like.config.num_hidden_layers
        
        self.arch = kwargs.pop('arch', 'BERT')
        self.freeze = kwargs.pop('freeze', True)
        
        self.from_tokenized = kwargs.pop('from_tokenized', True)
        self.pre_truncation = kwargs.pop('pre_truncation', False)
        self.agg_mode = kwargs.pop('agg_mode', 'mean')
        self.mix_layers = kwargs.pop('mix_layers', 'top')
        self.use_gamma = kwargs.pop('use_gamma', False)
        
        super().__init__(**kwargs)
        
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state['bert_like'] = None
        return state
        
        
    def _token_ids_from_string(self, raw_text: str):
        """
        Tokenize a non-tokenized sentence (string), and convert to sub-token indexes. 
        
        Parameters
        ----------
        raw_text : str
            e.g., "I like this movie."
            
        Returns
        -------
        sub_tok_ids: torch.LongTensor
            A 1D tensor of sub-token indexes.
        """
        sub_tokens = self.tokenizer.tokenize(raw_text)
        
        # Recommended by Sun et al. (2019)
        if len(sub_tokens) > (self.tokenizer.max_len - 2):
            head_len = self.tokenizer.max_len // 4
            tail_len = self.tokenizer.max_len - 2 - head_len
            sub_tokens = sub_tokens[:head_len] + sub_tokens[-tail_len:]
        
        sub_tok_ids = [self.tokenizer.cls_token_id] + \
                       self.tokenizer.convert_tokens_to_ids(sub_tokens) + \
                      [self.tokenizer.sep_token_id]
        # (step+2, )
        return torch.tensor(sub_tok_ids)
        
        
    def _token_ids_from_tokenized(self, tokenized_raw_text: List[str]):
        """
        Further tokenize a pre-tokenized sentence, and convert to sub-token indexes. 
        
        Parameters
        ----------
        tokenized_raw_text : List[str]
            e.g., ["I", "like", "this", "movie", "."]

        Returns
        -------
        sub_tok_ids: torch.LongTensor
            A 1D tensor of sub-token indexes.
        ori_indexes: torch.LongTensor
            A 1D tensor indicating each sub-token's original index in `tokenized_raw_text`.
        """
        nested_sub_tokens = [self.tokenizer.tokenize(word) for word in tokenized_raw_text]
        sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        
        # TODO: Sentence longer than 512?
        sub_tok_ids = [self.tokenizer.cls_token_id] + \
                       self.tokenizer.convert_tokens_to_ids(sub_tokens) + \
                      [self.tokenizer.sep_token_id]
        
        # (step+2, ), (step, )
        return torch.tensor(sub_tok_ids), torch.tensor(ori_indexes)
        
        
    def exemplify(self, tokens: TokenSequence):
        if self.from_tokenized:
            sub_tok_ids, ori_indexes = self._token_ids_from_tokenized(tokens.raw_text)
            return {'sub_tok_ids': sub_tok_ids, 
                    'ori_indexes': ori_indexes}
        else:
            # TODO:?
            sub_tok_ids = self._token_ids_from_string(" ".join(tokens.raw_text))
            return {'sub_tok_ids': sub_tok_ids}
            
        
    def batchify(self, batch_ex: List[dict]):
        batch_sub_tok_ids = [ex['sub_tok_ids'] for ex in batch_ex]
        sub_tok_seq_lens = torch.tensor([sub_tok_ids.size(0) for sub_tok_ids in batch_sub_tok_ids])
        batch_sub_tok_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_sub_tok_ids, 
                                                            batch_first=True, 
                                                            padding_value=self.tokenizer.pad_token_id)
        
        if self.from_tokenized:
            batch_ori_indexes = [ex['ori_indexes'] for ex in batch_ex]
            batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
            return {'sub_tok_ids': batch_sub_tok_ids, 
                    'sub_tok_mask': batch_sub_tok_mask, 
                    'ori_indexes': batch_ori_indexes}
        else:
            return {'sub_tok_ids': batch_sub_tok_ids, 
                    'sub_tok_mask': batch_sub_tok_mask}
        
    def instantiate(self):
        return BertLikeEmbedder(self)
    
    
    
class BertLikeEmbedder(torch.nn.Module):
    """
    An embedder based on BERT representations. 
    
    NOTE: No need to use `torch.no_grad()` if the `requires_grad` attributes of 
    the pretrained model have been properly set. 
    `torch.no_grad()` enforces the result of every computation in its context 
    to have `requires_grad=False`, even when the inputs have `requires_grad=True`.
    
    References
    ----------
    [1] C. Sun, et al. 2019. How to Fine-Tune BERT for Text Classification?
    """
    def __init__(self, config: BertLikeConfig):
        super().__init__()
        self.bert_like = config.bert_like
        
        self.from_tokenized = config.from_tokenized
        self.freeze = config.freeze
        self.mix_layers = config.mix_layers
        self.use_gamma = config.use_gamma
        
        if self.from_tokenized:
            self.group_aggregating = SequenceGroupAggregating(mode=config.agg_mode)
        if self.mix_layers.lower() == 'trainable':
            self.scalar_mix = ScalarMix(config.num_hidden_layers + 1)
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.tensor(1.0))
            
        # Register BERT configurations
        self.bert_like.requires_grad_(not self.freeze)
        
        
    def forward(self, 
                sub_tok_ids: torch.LongTensor, 
                sub_tok_mask: torch.BoolTensor, 
                ori_indexes: torch.LongTensor=None):
        # bert_outs: (batch, sub_tok_step+2, hid_dim)
        # hidden: list of (batch, sub_tok_step+2, hid_dim)
        bert_outs, _, hidden = self.bert_like(input_ids=sub_tok_ids, 
                                              attention_mask=(~sub_tok_mask).type(torch.long), 
                                              output_hidden_states=True)
        
        if self.mix_layers.lower() == 'trainable':
            bert_outs = self.scalar_mix(hidden)
        elif self.mix_layers.lower() == 'top':
            pass
        elif self.mix_layers.lower() == 'average':
            bert_outs = sum(hidden) / len(hidden)
        
        if self.use_gamma:
            bert_outs = self.gamma * bert_outs
        
        # Remove the `[CLS]` and `[SEP]` positions. 
        bert_outs = bert_outs[:, 1:-1]
        sub_tok_mask = sub_tok_mask[:, 2:]
        
        if self.from_tokenized:
            # bert_outs: (batch, tok_step, hid_dim)
            bert_outs = self.group_aggregating(bert_outs, ori_indexes)    
            
        return bert_outs
        
    