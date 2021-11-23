# -*- coding: utf-8 -*-
from typing import Union, List
import random
import torch
import transformers

from ..nn.functional import seq_lens2mask
from ..config import Config


class PreTrainingConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.stoi = self.tokenizer.get_vocab()
        self.special_ids = list(set(self.tokenizer.all_special_ids))
        self.non_special_ids = [idx for idx in self.stoi.values() if idx not in self.special_ids]
        
        super().__init__(**kwargs)
        
    @property
    def cls_id(self):
        return self.tokenizer.cls_token_id
        
    @property
    def sep_id(self):
        return self.tokenizer.sep_token_id
        
    @property
    def unk_id(self):
        return self.tokenizer.unk_token_id
        
    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id
        
    @property
    def mask_id(self):
        return self.tokenizer.mask_token_id



class MaskedLMConfig(PreTrainingConfig):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        
        self.masking_rate = kwargs.pop('masking_rate', 0.15)
        self.random_word_rate = kwargs.pop('random_word_rate', 0.1)
        self.unchange_rate = kwargs.pop('unchange_rate', 0.1)
        
        super().__init__(**kwargs)
        
        
    @property
    def mlm_label_mask_id(self):
        # transformers/models/bert/modeling_bert.py/BertForPreTraining
        # Tokens with indices set to `-100` are ignored (masked), the loss is only 
        # computed for the tokens with labels in `[0, ..., config.vocab_size]`
        return -100
        
        
    def exemplify(self, entry: Union[dict, str, List[str]], training: bool=True):
        """Use dynamic masking. 
        
        entry: dict / str / List[str]
            dict: {'tokens': TokenSequence, ...}
            str: A string, which is tokenized by `tokenizer` and re-joined with spaces. 
            List[str]: A list of string, which is tokenized by `tokenizer`.
        """
        if isinstance(entry, dict):
            tokenized_text = self.tokenizer.tokenize(" ".join(entry['tokens'].raw_text))
        elif isinstance(entry, str):
            # String inputs are always regarded as re-joined with spaces
            # Never pass raw, un-tokenized text here
            tokenized_text = entry.split(" ")
        else:
            assert isinstance(entry, list) and isinstance(entry[0], str)
            tokenized_text = entry
        
        mlm_tok_ids = [self.cls_id] + self.tokenizer.convert_tokens_to_ids(tokenized_text) + [self.sep_id]
        mlm_lab_ids = []
        for k, tok_id in enumerate(mlm_tok_ids):
            rv1, rv2 = random.random(), random.random()
            if (tok_id not in self.special_ids) and (rv1 < self.masking_rate):
                if rv2 < (1 - self.random_word_rate - self.unchange_rate):
                    mlm_tok_ids[k] = self.mask_id  # Replace with `[MASK]`
                elif rv2 < (1 - self.unchange_rate):
                    mlm_tok_ids[k] = random.choice(self.non_special_ids)  # Replace with a random token
                else:
                    pass  # Keep the original token unchange
                mlm_lab_ids.append(tok_id)
            else:
                mlm_lab_ids.append(self.mlm_label_mask_id)
        
        return {'mlm_tok_ids': torch.tensor(mlm_tok_ids), 
                'mlm_lab_ids': torch.tensor(mlm_lab_ids)}
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_mlm_tok_ids = [ex['mlm_tok_ids'] for ex in batch_ex]
        batch_mlm_lab_ids = [ex['mlm_lab_ids'] for ex in batch_ex]
        
        mlm_tok_seq_lens = torch.tensor([s.size(0) for s in batch_mlm_tok_ids])
        mlm_att_mask = seq_lens2mask(mlm_tok_seq_lens)
        batch_mlm_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_tok_ids, batch_first=True, padding_value=self.pad_id)
        batch_mlm_lab_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_lab_ids, batch_first=True, padding_value=self.mlm_label_mask_id)
        
        return {'mlm_tok_ids': batch_mlm_tok_ids, 
                'mlm_lab_ids': batch_mlm_lab_ids, 
                'mlm_att_mask': mlm_att_mask}
        
        
    def instantiate(self):
        return self.bert_like
