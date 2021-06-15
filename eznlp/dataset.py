# -*- coding: utf-8 -*-
from typing import List
import torch

from .nn.functional import seq_lens2mask
from .wrapper import Batch
from .model.model import ModelConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfig=None, training: bool=True):
        """
        Parameters
        ----------
        data : List[dict]
            Each entry (as a dict) follows the format of:
                {'tokens': TokenSequence, 'label': str, 'chunks': List[tuple], 'relations': List[tuple], ...}
            where (1) `label` is a str (or int).  
                  (2) each `chunk` follows the format of (chunk_type, chunk_start, chunk_end). 
                  (3) each `relation` follows the format of (relation_type, head_chunk, tail_chunk), 
                      i.e., (relation_type, (head_type, head_start, head_end), (tail_type, tail_start, tail_end)). 
            
        """
        super().__init__()
        self.data = data
        self.config = config
        self.training = training
        
    def __len__(self):
        return len(self.data)
        
    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")
        
        if 'raw_idx' in self.data[0]:
            num_raws = len({data_entry['raw_idx'] for data_entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")
            
        seq_lens = [len(data_entry['tokens']) for data_entry in self.data]
        ave_len = sum(seq_lens) / len(seq_lens)
        max_len = max(seq_lens)
        summary.append(f"The average sequence length is {ave_len:,.1f}")
        summary.append(f"The maximum sequence length is {max_len:,}")
        
        if 'label' in self.data[0]:
            num_label_types = len({data_entry['label'] for data_entry in self.data})
            summary.append(f"The dataset has {num_label_types:,} categories")
        
        if 'chunks' in self.data[0]:
            num_chunks = sum(len(data_entry['chunks']) for data_entry in self.data)
            num_chunk_types = len({ck[0] for data_entry in self.data for ck in data_entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")
        
        if 'attributes' in self.data[0]:
            num_attributes = sum(len(data_entry['attributes']) for data_entry in self.data)
            num_attr_types = len({attr[0] for data_entry in self.data for attr in data_entry['attributes']})
            summary.append(f"The dataset has {num_attributes:,} attributes of {num_attr_types:,} types")
        
        if 'relations' in self.data[0]:
            num_relations = sum(len(data_entry['relations']) for data_entry in self.data)
            num_relation_types = len({rel[0] for data_entry in self.data for rel in data_entry['relations']})
            summary.append(f"The dataset has {num_relations:,} relations of {num_relation_types:,} types")
        
        return "\n".join(summary)
        
        
    def build_vocabs_and_dims(self, *others):
        self.config.build_vocabs_and_dims(self.data, *others)
        
        
    def __getitem__(self, i):
        data_entry = self.data[i]
        example = {'tokenized_text': data_entry['tokens'].text}
        
        example.update(self.config.exemplify(data_entry, training=self.training))
        return example
    
    
    def collate(self, batch_examples: List[dict]):
        batch = {}
        batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
        batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
        batch['mask'] = seq_lens2mask(batch['seq_lens'])
        
        batch.update(self.config.batchify(batch_examples))
        return Batch(**batch)
