# -*- coding: utf-8 -*-
import transformers

from ..config import Config


class PreTrainingConfig(Config):
    """Configurations for LM pretraining, e.g., masked LM, left-to-right LM. 
    
    """
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
