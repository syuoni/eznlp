# -*- coding: utf-8 -*-
from typing import List
import tqdm
import numpy as np

from ..data.dataset import Dataset
from .classifier import TextClassifierConfig
from ..sequence_tagging.transition import find_ascending


class TextClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: TextClassifierConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 'label': str/int}, ...]
            
            `label` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = TextClassifierConfig()
        super().__init__(data, config)
        
    def _truncate_tokens(self):
        tokenizer = self.config.bert_like.tokenizer
        max_len = tokenizer.max_len - 2
        head_len = tokenizer.max_len // 4
        tail_len = max_len - head_len
        
        for data_entry in tqdm.tqdm(self.data):
            tokens = data_entry['tokens']
            nested_sub_tokens = [tokenizer.tokenize(word) for word in tokens.raw_text]
            sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]
            
            if sum(sub_tok_seq_lens) > max_len:
                cum_lens = np.cumsum(sub_tok_seq_lens).tolist()
                find_head, head_end = find_ascending(cum_lens, head_len)
                if find_head:
                    head_end += 1
                    
                rev_cum_lens = np.cumsum(sub_tok_seq_lens[::-1]).tolist()
                find_tail, tail_begin = find_ascending(rev_cum_lens, tail_len)
                if find_tail:
                    tail_begin += 1
                    
                data_entry['tokens'] = tokens[:head_end] + tokens[-tail_begin:]
                
    @property
    def summary(self):
        summary = [super().summary]
        
        n_labels = len(self.config.decoder.label2idx)
        summary.append(f"The dataset has {n_labels:,} labels")
        return "\n".join(summary)
    
    