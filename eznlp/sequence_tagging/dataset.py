# -*- coding: utf-8 -*-
from typing import List

from ..data.dataset import Dataset
from .tagger import SequenceTaggerConfig


class SequenceTaggingDataset(Dataset):
    def __init__(self, data: List[dict], config: SequenceTaggerConfig=None, training: bool=True):
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
        super().__init__(data, config, training=training)
        
    @property
    def summary(self):
        summary = [super().summary]
        
        num_chunks = sum(len(data_entry['chunks']) for data_entry in self.data)
        summary.append(f"The dataset has {num_chunks:,} chunks")
        num_tags = len(self.config.decoder.tag2idx)
        summary.append(f"The dataset has {num_tags:,} tag-types")
        return "\n".join(summary)
