# -*- coding: utf-8 -*-
from typing import List
import random
import logging
import torch

from ..wrapper import Batch
from ..dataset import Dataset
from .model import MaskedLMConfig
from ..model.bert_like import _tokenized2nested

logger = logging.getLogger(__name__)



class MaskedLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling. 
    """
    def __init__(self, data: List[dict], config: MaskedLMConfig, training: bool=True):
        super().__init__(data, config, training=training)
        
    def __getitem__(self, i):
        data_entry = self.data[i]
        
        nested_sub_tokens = _tokenized2nested(data_entry['tokens'].raw_text, self.config.tokenizer)
        sub_tokens = [sub_tok for i, tok in enumerate(nested_sub_tokens) for sub_tok in tok]
        return self.config.exemplify(sub_tokens)
        
    
    def collate(self, batch_examples: List[dict]):
        batch = self.config.batchify(batch_examples)
        return Batch(**batch)
    
    
    
def _slice_chunk(chunk_id, num_chunks, num_items):
    assert chunk_id >= 0 and num_chunks > 0 and chunk_id < num_chunks
    
    chunk_size = num_items / num_chunks
    start = int(chunk_size* chunk_id    + 0.5)
    end   = int(chunk_size*(chunk_id+1) + 0.5)
    return slice(start, end)


class FolderLikeMaskedLMDataset(torch.utils.data.IterableDataset):
    """
    Folder-like Dataset for Masked Language Modeling. 
    """
    def __init__(self, text_paths: List[str], config: MaskedLMConfig, 
                 max_len=512, shuffle=True, mp_rank=0, mp_world_size=0):
        super().__init__()
        if mp_rank >= 0 and mp_world_size > 0 and mp_rank < mp_world_size:
            text_paths = text_paths[_slice_chunk(mp_rank, mp_world_size, len(text_paths))]
        logger.info(f"Totally {len(text_paths)} text files in the {mp_rank}-th process")
        
        self.text_paths = text_paths
        self.config = config
        self.max_len = max_len
        self.shuffle = shuffle
        
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the full iterator
            this_text_paths = self.text_paths[:]
        else:
            # in a worker process -> split workload
            this_text_paths = self.text_paths[_slice_chunk(worker_info.id, 
                                                           worker_info.num_workers, 
                                                           len(self.text_paths))]
            
        if self.shuffle:
            random.shuffle(this_text_paths)
            
        cut_len = self.max_len - 2
        for text_path in this_text_paths:
            try:
                with open(text_path, encoding='utf-8') as f:
                    text = f.read()
            except:
                continue
            
            sub_tokens_from_text = self.config.tokenizer.tokenize(text)
            num_examples = len(sub_tokens_from_text) // cut_len
            if len(sub_tokens_from_text) % cut_len >= (cut_len / 10):
                num_examples += 1
            
            for k in range(num_examples):
                sub_tokens = sub_tokens_from_text[(cut_len*k):(cut_len*(k+1))]
                yield self.config.exemplify(sub_tokens)
                
                
    def collate(self, batch_examples: List[str]):
        batch = self.config.batchify(batch_examples)
        return Batch(**batch)
    
    