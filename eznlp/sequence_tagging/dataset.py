# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch

from ..data import Dataset, TensorWrapper
from .decoder import DecoderConfig
from .tagger import SequenceTaggerConfig
from .transition import ChunksTagsTranslator


class SequenceTaggingDataset(Dataset):
    def __init__(self, data: List[dict], config: SequenceTaggerConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 
              'chunks': list of tuple}, ...]
            
            `chunks` should be [(chunk_type, chunk_start, chunk_end), ...].
            `chunks` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = SequenceTaggerConfig()
        super().__init__(data, config)
        
        self.translator = ChunksTagsTranslator(scheme=self.config.decoder.scheme)
        self._is_labeled = ('chunks' in data[0])
        if self._is_labeled:
            self._build_tags()
        
        self._building_vocabs = (self.config.decoder.idx2tag is None)
        if self._building_vocabs:
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
                
                
    def _build_tags(self):
        for curr_data in self.data:
            curr_data['tags'] = self.translator.chunks2tags(curr_data['chunks'], len(curr_data['tokens']))
            
            
    def _build_tag_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            counter.update(curr_data['tags'])
        self.config.decoder.set_vocab(idx2tag=['<pad>'] + list(counter.keys()))
            
    
    def _check_tag_vocab(self):
        counter = Counter()
        for curr_data in self.data:
            counter.update(curr_data['tags'])
            
        oov_tags = [tag for tag in counter if tag not in self.config.decoder.tag2idx]
        if len(oov_tags) > 0:
            raise RuntimeError(f"OOV tags exist: {oov_tags}")
            
    # TODO
    # def _build_cas_tag_vocab(self):
    #     cas_tag_counter = Counter()
    #     cas_type_counter = Counter()
    #     for curr_data in self.data:
    #         cas_tag_counter.update(self.config.decoder.build_cas_tags_by_tags(curr_data['tags']))
    #         cas_type_counter.update(self.config.decoder.build_cas_types_by_tags(curr_data['tags']))
            
    #     self.config.decoder.set_vocabs(idx2cas_tag=['<pad>'] + list(cas_tag_counter.keys()), 
    #                                    idx2cas_type=['<pad>'] + list(cas_type_counter.keys()))
        
    def extra_summary(self):
        n_chunks = sum(len(curr_data['chunks']) for curr_data in self.data)
        return f"The dataset has {n_chunks:,} chunks"
        
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        example = self._get_basic_example(curr_data)
        
        tags_obj = Tags(curr_data['tags'], self.config.decoder) if self._is_labeled else None
        example.add_attributes(tags_obj=tags_obj)
        return example
    
    
    def collate(self, batch_examples):
        batch = self._build_basic_batch(batch_examples)
        
        batch_tags_objs = [ex.tags_obj for ex in batch_examples] if self._is_labeled else None
        batch.add_attributes(tags_objs=batch_tags_objs)
        return batch
    
    
class Tags(TensorWrapper):
    def __init__(self, tags: list, config: DecoderConfig):
        self.tags = tags
        # self.cas_tags = config.build_cas_tags_by_tags(tags)
        # self.cas_types = config.build_cas_types_by_tags(tags)
        # self.cas_ent_slices, self.cas_ent_types = config.build_cas_ent_slices_and_types_by_tags(tags)
        
        self.tag_ids = torch.tensor(config.tags2ids(self.tags))
        # self.cas_tag_ids = torch.tensor(config.cas_tags2ids(self.cas_tags))
        # self.cas_type_ids = torch.tensor(config.cas_types2ids(self.cas_types))
        # self.cas_ent_type_ids = torch.tensor(config.cas_types2ids(self.cas_ent_types))
        
    def _apply_to_tensors(self, func):
        self.tag_ids = func(self.tag_ids)
        # self.cas_tag_ids = func(self.cas_tag_ids)
        # self.cas_type_ids = func(self.cas_type_ids)
        # self.cas_ent_type_ids = func(self.cas_ent_type_ids)
        return self
    
    