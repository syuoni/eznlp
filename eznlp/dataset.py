# -*- coding: utf-8 -*-
from typing import List
import torch

from .nn.functional import seq_lens2mask
from .wrapper import Batch
from .model.model import ModelConfigBase


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase=None, training: bool=True):
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
    def _has_tokens(self):
        return 'tokens' in self.data[0]
        
    def _make_tokens_summary(self, field: str='tokens'):
        seq_lens = [len(entry[field]) for entry in self.data]
        ave_len = sum(seq_lens) / len(seq_lens)
        max_len = max(seq_lens)
        return [f"The average `{field}` length is {ave_len:,.1f}", 
                f"The maximum `{field}` length is {max_len:,}"]
        
    @property
    def summary(self):
        summary = []
        num_seqs = len(self.data)
        summary.append(f"The dataset consists {num_seqs:,} sequences")
        
        if 'raw_idx' in self.data[0]:
            num_raws = len({entry['raw_idx'] for entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")
        
        if 'tokens' in self.data[0]:
            summary.extend(self._make_tokens_summary('tokens'))
        
        if 'trg_tokens' in self.data[0]:
            summary.extend(self._make_tokens_summary('trg_tokens'))
        
        if 'label' in self.data[0]:
            num_label_types = len({entry['label'] for entry in self.data})
            summary.append(f"The dataset has {num_label_types:,} categories")
        
        if 'chunks' in self.data[0]:
            num_chunks = sum(len(entry['chunks']) for entry in self.data)
            num_chunk_types = len({ck[0] for entry in self.data for ck in entry['chunks']})
            summary.append(f"The dataset has {num_chunks:,} chunks of {num_chunk_types:,} types")
        
        if 'attributes' in self.data[0]:
            num_attributes = sum(len(entry['attributes']) for entry in self.data)
            num_attr_types = len({attr[0] for entry in self.data for attr in entry['attributes']})
            summary.append(f"The dataset has {num_attributes:,} attributes of {num_attr_types:,} types")
        
        if 'relations' in self.data[0]:
            num_relations = sum(len(entry['relations']) for entry in self.data)
            num_relation_types = len({rel[0] for entry in self.data for rel in entry['relations']})
            summary.append(f"The dataset has {num_relations:,} relations of {num_relation_types:,} types")
        
        return "\n".join(summary)
        
        
    def build_vocabs_and_dims(self, *others):
        self.config.build_vocabs_and_dims(self.data, *others)
        
        
    def __getitem__(self, i):
        entry = self.data[i]
        example = {}
        if self._has_tokens:
            example['tokenized_text'] = entry['tokens'].text
        
        example.update(self.config.exemplify(entry, training=self.training))
        return example
        
        
    def collate(self, batch_examples: List[dict]):
        batch = {}
        if self._has_tokens:
            batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
            batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
            batch['mask'] = seq_lens2mask(batch['seq_lens'])
        
        batch.update(self.config.batchify(batch_examples))
        return Batch(**batch)
