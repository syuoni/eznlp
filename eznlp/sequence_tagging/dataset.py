# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch

from ..data import Dataset, TensorWrapper
from .decoder import SequenceTaggingDecoderConfig
from .tagger import SequenceTaggerConfig


class SequenceTaggingDataset(Dataset):
    def __init__(self, data: List[dict], config: SequenceTaggerConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 
              'chunks': list}, ...]
            
            `chunks` should be [(chunk_type, chunk_start, chunk_end), ...].
            `chunks` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = SequenceTaggerConfig()
        super().__init__(data, config)
        
        self._is_labeled = ('chunks' in data[0])
        self._is_building_vocabs = (self.config.decoder.idx2tag is None)
        
        if self._is_building_vocabs:
            assert self._is_labeled
            self._build_token_vocab()
            self._build_tag_vocab()
            
            if self.config.embedder.char is not None:
                self._build_char_vocab()
                
            if self.config.embedder.enum is not None:
                self._build_enum_vocabs()
                
            self.config._update_dims(self.data[0]['tokens'][0])
        else:
            if self._is_labeled:
                self._check_tag_vocab()
                
                
    def _build_tag_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            curr_tags = self.config.decoder.translator.chunks2tags(curr_data['chunks'], len(curr_data['tokens']))
            counter.update(curr_tags)
        self.config.decoder.idx2tag = ['<pad>'] + list(counter.keys())
        
        
    def _check_tag_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            curr_tags = self.config.decoder.translator.chunks2tags(curr_data['chunks'], len(curr_data['tokens']))
            counter.update(curr_tags)
            
        oov = [tag for tag in counter if tag not in self.config.decoder.tag2idx]
        if len(oov) > 0:
            raise RuntimeError(f"OOV tags exist: {oov}")
            
            
    @property
    def extra_summary(self):
        n_chunks = sum(len(curr_data['chunks']) for curr_data in self.data)
        return f"The dataset has {n_chunks:,} chunks"
        
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        example = self._get_basic_example(curr_data)
        
        tags_obj = Tags(curr_data, self.config.decoder) if self._is_labeled else None
        example.add_attributes(tags_obj=tags_obj)
        return example
    
    
    def collate(self, batch_examples):
        batch = self._build_basic_batch(batch_examples)
        
        batch_tags_objs = [ex.tags_obj for ex in batch_examples] if self._is_labeled else None
        batch.add_attributes(tags_objs=batch_tags_objs)
        return batch
    
    
class Tags(TensorWrapper):
    """
    Packaging of tags and chunks. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': list}
    """
    def __init__(self, data_entry: dict, config: SequenceTaggingDecoderConfig):
        self.chunks = data_entry['chunks']
        self.tags = config.translator.chunks2tags(data_entry['chunks'], len(data_entry['tokens']))
        self.tag_ids = torch.tensor([config.tag2idx[t] for t in self.tags], dtype=torch.long)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.tags})"
        
    