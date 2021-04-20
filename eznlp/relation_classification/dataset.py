# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .classifier import RelationClassifierConfig


class RelationClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: RelationClassifierConfig=None, neg_sampling=True):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 'chunks': list, 'relations': list}, ...]
            
            `relations` should be [(relation_type, 
                                    (head_type, head_start, head_end), 
                                    (tail_type, tail_start, tail_end)), ...].
            `relations` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = RelationClassifierConfig()
        super().__init__(data, config)
        self.neg_sampling = neg_sampling
        
    @property
    def summary(self):
        summary = [super().summary]
        
        n_chunks = sum(len(data_entry['chunks']) for data_entry in self.data)
        summary.append(f"The dataset has {n_chunks:,} chunks")
        n_relations = sum(len(data_entry['relations']) for data_entry in self.data)
        summary.append(f"The dataset has {n_relations:,} relations")
        n_labels = len(self.config.decoder.label2idx)
        summary.append(f"The dataset has {n_labels:,} labels")
        return "\n".join(summary)
    
    def __getitem__(self, i):
        data_entry = self.data[i]
        example = {'tokenized_text': data_entry['tokens'].text}
        
        example.update(self.config.exemplify(data_entry, neg_sampling=self.neg_sampling))
        return example
        