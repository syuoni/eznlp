# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch

from ..data import Dataset
from .classifier import TextClassifierConfig


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
    
    