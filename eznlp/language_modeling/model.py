# -*- coding: utf-8 -*-
from typing import List
import random
import torch
import transformers

from ..nn.functional import seq_lens2mask
from ..config import Config


class MaskedLMConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.mask_id = self.tokenizer.mask_token_id
        
        self.stoi = self.tokenizer.get_vocab()
        self.special_ids = list(set(self.tokenizer.all_special_ids))
        self.non_special_ids = [idx for idx in self.stoi.values() if idx not in self.special_ids]
        
        # Tokens with indices set to `-100` are ignored (masked), the loss is only 
        # computed for the tokens with labels in `[0, ..., config.vocab_size]`
        self.mlm_label_mask_id = -100
        
        self.masking_prob = kwargs.pop('masking_prob', 0.15)
        self.random_word_prob = kwargs.pop('random_word_prob', 0.1)
        self.unchange_prob = kwargs.pop('unchange_prob', 0.1)
        
        super().__init__(**kwargs)
        
        
    def exemplify(self, sub_tokens: List[str]):
        """Dynamic Masking. 
        """
        mlm_tok_ids = [self.cls_id] + [self.stoi[tok] for tok in sub_tokens] + [self.sep_id]
        mlm_lab_ids = []
        for k, tok_id in enumerate(mlm_tok_ids):
            rv1, rv2 = random.random(), random.random()
            if (tok_id not in self.special_ids) and (rv1 < self.masking_prob):
                if rv2 < (1 - self.random_word_prob - self.unchange_prob):
                    # Replace with `[MASK]`
                    mlm_tok_ids[k] = self.mask_id
                elif rv2 < (1 - self.unchange_prob):
                    # Replace with a random token
                    mlm_tok_ids[k] = random.choice(self.non_special_ids)
                else:
                    # Keep the original token unchange
                    pass
                mlm_lab_ids.append(tok_id)
            else:
                mlm_lab_ids.append(self.mlm_label_mask_id)
                
        return {'mlm_tok_ids': torch.tensor(mlm_tok_ids), 
                'mlm_lab_ids': torch.tensor(mlm_lab_ids)}

    
    def batchify(self, batch_ex: List[dict]):
        batch_mlm_tok_ids = [ex['mlm_tok_ids'] for ex in batch_ex]
        batch_mlm_lab_ids = [ex['mlm_lab_ids'] for ex in batch_ex]
        
        mlm_tok_seq_lens = torch.tensor([s.size(0) for s in batch_mlm_tok_ids])
        attention_mask = seq_lens2mask(mlm_tok_seq_lens)
        batch_mlm_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_tok_ids, batch_first=True, padding_value=self.pad_id)
        batch_mlm_lab_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_lab_ids, batch_first=True, padding_value=self.mlm_label_mask_id)
        
        return {'mlm_tok_ids': batch_mlm_tok_ids, 
                'mlm_lab_ids': batch_mlm_lab_ids, 
                'attention_mask': attention_mask}
    
    