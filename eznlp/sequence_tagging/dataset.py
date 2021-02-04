# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .tagger import SequenceTaggerConfig


class SequenceTaggingDataset(Dataset):
    def __init__(self, data: List[dict], config: SequenceTaggerConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 'chunks': list}, ...]
            
            `chunks` should be [(chunk_type, chunk_start, chunk_end), ...].
            `chunks` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = SequenceTaggerConfig()
        super().__init__(data, config)
        
    @property
    def summary(self):
        summary = [super().summary]
        
        n_chunks = sum(len(curr_data['chunks']) for curr_data in self.data)
        summary.append(f"The dataset has {n_chunks:,} chunks")
        return "\n".join(summary)
        
    