# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .classifier import RelationClassifierConfig


class RelationClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: RelationClassifierConfig=None, training: bool=True):
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
        super().__init__(data, config, training=training)
        
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

