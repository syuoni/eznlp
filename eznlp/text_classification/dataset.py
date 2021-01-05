# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import tqdm
import torch
import numpy as np

from ..data import Dataset
from .classifier import TextClassifierConfig
from ..sequence_tagging.transition import find_ascending


class TextClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: TextClassifierConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 
              'label': str/int}, ...]
            
            `label` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = TextClassifierConfig()
        super().__init__(data, config)
        
        self._is_labeled = ('label' in data[0])
        
        if self.config.bert_like_embedder is not None and self.config.bert_like_embedder.pre_truncation:
            self._truncate_tokens()
            
        self._building_vocabs = (self.config.decoder.idx2label is None)
        if self._building_vocabs:
            assert self._is_labeled
            
            self._build_token_vocab()
            self._build_label_vocab()
            
            if self.config.embedder.char is not None:
                self._build_char_vocab()
            
            if self.config.embedder.enum is not None:
                self._build_enum_vocabs()
                
            self.config._update_dims(self.data[0]['tokens'][0])
        else:
            if self._is_labeled:
                self._check_label_vocab()
                
                
    def _truncate_tokens(self):
        tokenizer = self.config.bert_like_embedder.tokenizer
        max_len = tokenizer.max_len - 2
        head_len = tokenizer.max_len // 4
        tail_len = max_len - head_len
        
        for curr_data in tqdm.tqdm(self.data):
            tokens = curr_data['tokens']
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
                    
                curr_data['tokens'] = tokens[:head_end] + tokens[-tail_begin:]
                
                
    def _build_label_vocab(self):
        counter = Counter([curr_data['label'] for curr_data in self.data])
        self.config.decoder.set_vocab(idx2label=list(counter.keys()))
        
    def _check_label_vocab(self):
        counter = Counter([curr_data['label'] for curr_data in self.data])
        
        oov = [label for label in counter if label not in self.config.decoder.label2idx]
        if len(oov) > 0:
            raise RuntimeError(f"OOV labels exist: {oov}")
            
    @property
    def extra_summary(self):
        n_labels = len(self.config.decoder.label2idx)
        return f"The dataset has {n_labels:,} labels"
        
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        example = self._get_basic_example(curr_data)
        
        label_id = torch.tensor(self.config.decoder.label2idx[curr_data['label']]) if self._is_labeled else None
        example.add_attributes(label_id=label_id)
        return example
    
    
    def collate(self, batch_examples):
        batch = self._build_basic_batch(batch_examples)
        
        batch_label_id = torch.stack([ex.label_id for ex in batch_examples]) if self._is_labeled else None
        batch.add_attributes(label_id=batch_label_id)
        return batch
    
    