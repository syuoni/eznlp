# -*- coding: utf-8 -*-
from typing import List
import torch

from ..nn.functional import seq_lens2mask
from ..config import Config
from .wrapper import Batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: Config, training: bool=True):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, ...}, ...]
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
        n_seqs = len(self.data)
        summary.append(f"The dataset consists {n_seqs:,} sequences")
        
        if 'raw_idx' in self.data[0]:
            n_raws = len({data_entry['raw_idx'] for data_entry in self.data})
            summary.append(f"\tbuilt from {n_raws:,} raw entries")
            
        seq_lens = [len(data_entry['tokens']) for data_entry in self.data]
        ave_len = sum(seq_lens) / len(seq_lens)
        max_len = max(seq_lens)
        summary.append(f"The average sequence length is {ave_len:,.1f}")
        summary.append(f"The max sequence length is {max_len:,}")
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
        