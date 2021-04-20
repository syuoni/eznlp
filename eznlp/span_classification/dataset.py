# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .classifier import SpanClassifierConfig


class SpanClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: SpanClassifierConfig=None, neg_sampling=True):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 'label': str/int}, ...]
            
            `label` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = SpanClassifierConfig()
        super().__init__(data, config)
        self.neg_sampling = neg_sampling
        
    @property
    def summary(self):
        summary = [super().summary]
        
        n_labels = len(self.config.decoder.label2idx)
        summary.append(f"The dataset has {n_labels:,} labels")
        return "\n".join(summary)
    
    def __getitem__(self, i):
        data_entry = self.data[i]
        example = {'tokenized_text': data_entry['tokens'].text}
        
        example.update(self.config.exemplify(data_entry, neg_sampling=self.neg_sampling))
        return example
        