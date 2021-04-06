# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .classifier import TextClassifierConfig


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
        
    @property
    def summary(self):
        summary = [super().summary]
        
        n_labels = len(self.config.decoder.label2idx)
        summary.append(f"The dataset has {n_labels:,} labels")
        return "\n".join(summary)
    
    
    

    
