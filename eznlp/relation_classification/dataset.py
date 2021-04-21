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
        
        num_chunks = sum(len(data_entry['chunks']) for data_entry in self.data)
        summary.append(f"The dataset has {num_chunks:,} chunks")
        num_ck_labels = len(self.config.decoder.ck_label2idx)
        summary.append(f"The dataset has {num_ck_labels:,} chunk-types")
        
        num_relations = sum(len(data_entry['relations']) for data_entry in self.data)
        summary.append(f"The dataset has {num_relations:,} relations")
        num_rel_labels = len(self.config.decoder.rel_label2idx)
        summary.append(f"The dataset has {num_rel_labels:,} relation-types")
        return "\n".join(summary)

