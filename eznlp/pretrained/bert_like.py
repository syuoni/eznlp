# -*- coding: utf-8 -*-
from typing import List
import re
import truecase
import torch
import transformers

from ..token import TokenSequence
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
        self.use_truecase = kwargs.pop('use_truecase', False)
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
        # Sequence longer than maximum length should be pre-processed
        assert len(sub_tokens) <= self.tokenizer.model_max_length - 2
        
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
        # The tokenizer returns an empty list if the input is a space-like string
        nested_sub_tokens = [word if len(word) > 0 else [self.tokenizer.unk_token] for word in nested_sub_tokens]
        sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        ori_indexes = [i for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        # Sequence longer than maximum length should be pre-processed
        assert len(sub_tokens) <= self.tokenizer.model_max_length - 2
        
        sub_tok_ids = [self.tokenizer.cls_token_id] + \
                       self.tokenizer.convert_tokens_to_ids(sub_tokens) + \
                      [self.tokenizer.sep_token_id]
        
        # (step+2, ), (step, )
        return torch.tensor(sub_tok_ids), torch.tensor(ori_indexes)
        
        
    def exemplify(self, tokens: TokenSequence):
        tokenized_raw_text = tokens.raw_text
        if self.use_truecase:
            tokenized_raw_text = _truecase(tokenized_raw_text)
            
        if self.from_tokenized:
            sub_tok_ids, ori_indexes = self._token_ids_from_tokenized(tokenized_raw_text)
            return {'sub_tok_ids': sub_tok_ids, 
                    'ori_indexes': ori_indexes}
        else:
            # Use rejoined tokenized raw text here
            sub_tok_ids = self._token_ids_from_string(" ".join(tokenized_raw_text))
            return {'sub_tok_ids': sub_tok_ids}
            
        
    def batchify(self, batch_ex: List[dict]):
        batch_sub_tok_ids = [ex['sub_tok_ids'] for ex in batch_ex]
        sub_tok_seq_lens = torch.tensor([sub_tok_ids.size(0) for sub_tok_ids in batch_sub_tok_ids])
        batch_sub_mask = seq_lens2mask(sub_tok_seq_lens)
        batch_sub_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_sub_tok_ids, 
                                                            batch_first=True, 
                                                            padding_value=self.tokenizer.pad_token_id)
        
        if self.from_tokenized:
            batch_ori_indexes = [ex['ori_indexes'] for ex in batch_ex]
            batch_ori_indexes = torch.nn.utils.rnn.pad_sequence(batch_ori_indexes, batch_first=True, padding_value=-1)
            return {'sub_tok_ids': batch_sub_tok_ids, 
                    'sub_mask': batch_sub_mask, 
                    'ori_indexes': batch_ori_indexes}
        else:
            return {'sub_tok_ids': batch_sub_tok_ids, 
                    'sub_mask': batch_sub_mask}
        
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
                sub_mask: torch.BoolTensor, 
                ori_indexes: torch.LongTensor=None):
        # last_hidden: (batch, sub_tok_step+2, hid_dim)
        # pooler_output: (batch, hid_dim)
        # hidden: a tuple of (batch, sub_tok_step+2, hid_dim)
        bert_outs = self.bert_like(input_ids=sub_tok_ids, 
                                   attention_mask=(~sub_mask).type(torch.long), 
                                   output_hidden_states=True)
        bert_hidden = bert_outs['hidden_states']
        
        # bert_hidden: (batch, sub_tok_step+2, hid_dim)
        if self.mix_layers.lower() == 'trainable':
            bert_hidden = self.scalar_mix(bert_hidden)
        elif self.mix_layers.lower() == 'top':
            bert_hidden = bert_hidden[-1]
        elif self.mix_layers.lower() == 'average':
            bert_hidden = sum(bert_hidden) / len(bert_hidden)
        
        if self.use_gamma:
            bert_hidden = self.gamma * bert_hidden
        
        # Remove the `[CLS]` and `[SEP]` positions. 
        bert_hidden = bert_hidden[:, 1:-1]
        sub_mask = sub_mask[:, 2:]
        
        if self.from_tokenized:
            # bert_hidden: (batch, tok_step, hid_dim)
            bert_hidden = self.group_aggregating(bert_hidden, ori_indexes)
            
        return bert_hidden
    
    
    
def _truecase(tokenized_raw_text: List[str]):
    """
    Get the truecased text. 
    
    Original:  ['FULL', 'FEES', '1.875', 'REOFFER', '99.32', 'SPREAD', '+20', 'BP']
    Truecased: ['Full', 'fees', '1.875', 'Reoffer', '99.32', 'spread', '+20', 'BP']
    
    References
    ----------
    [1] https://github.com/google-research/bert/issues/223
    [2] https://github.com/daltonfury42/truecase
    """
    new_tokenized = tokenized_raw_text.copy()
    
    word_lst = [(w, idx) for idx, w in enumerate(new_tokenized) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]
    
    if len(lst) > 0 and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()

        # the trucaser have its own tokenization ...
        # skip if the number of word dosen't match
        if len(parts) == len(word_lst): 
            for (w, idx), nw in zip(word_lst, parts):
                new_tokenized[idx] = nw
                
    return new_tokenized
    
    