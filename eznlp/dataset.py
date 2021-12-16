# -*- coding: utf-8 -*-
from typing import List, Any
import random
import torch

from .nn.functional import seq_lens2mask
from .wrapper import Batch
from .model.model import ModelConfigBase
from .plm import PreTrainingConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase, training: bool=True):
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
            num_raws = len({entry['raw_idx'] for entry in self.data})
            summary.append(f"\tbuilt from {num_raws:,} raw entries")
        
        if 'tokens' in self.data[0]:
            seq_lens = [len(entry['tokens']) for entry in self.data]
            ave_len, max_len = sum(seq_lens)/len(seq_lens), max(seq_lens)
            summary.extend([f"The average `tokens` length is {ave_len:,.1f}", 
                            f"The maximum `tokens` length is {max_len:,}"])
        
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
        
    def _get_entry(self, i):
        return self.data[i]
        
    def __getitem__(self, i):
        entry = self._get_entry(i)
        example = {}
        if 'tokens' in self.data[0]:
            example['tokenized_text'] = entry['tokens'].text
        
        example.update(self.config.exemplify(entry, training=self.training))
        return example
        
        
    def collate(self, batch_examples: List[dict]):
        batch = {}
        if 'tokens' in self.data[0]:
            batch['tokenized_text'] = [ex['tokenized_text'] for ex in batch_examples]
            batch['seq_lens'] = torch.tensor([len(tokenized_text) for tokenized_text in batch['tokenized_text']])
            batch['mask'] = seq_lens2mask(batch['seq_lens'])
        
        batch.update(self.config.batchify(batch_examples))
        return Batch(**batch)




class GenerationDataset(Dataset):
    def __init__(self, data: List[dict], config: ModelConfigBase=None, training: bool=True):
        super().__init__(data, config=config, training=training)
        if training:
            self._indexing = [(src_idx, trg_idx) for src_idx, entry in enumerate(self.data) 
                                for trg_idx, tokens in enumerate(entry['full_trg_tokens'])]
        
    @property
    def summary(self):
        summary = [super().summary]
        
        seq_lens = [len(tokens) for entry in self.data for tokens in entry['full_trg_tokens']]
        ave_len, max_len = sum(seq_lens)/len(seq_lens), max(seq_lens)
        summary.extend([f"The average `trg_tokens` length is {ave_len:,.1f}", 
                        f"The maximum `trg_tokens` length is {max_len:,}"])
        return "\n".join(summary)
        
        
    def __len__(self):
        if self.training:
            return len(self._indexing)
        else:
            return len(self.data)
        
    def _get_entry(self, i):
        if self.training:
            src_idx, trg_idx = self._indexing[i]
            entry = self.data[src_idx]
            # `trg_tokens` is a cache field
            entry['trg_tokens'] = entry['full_trg_tokens'][trg_idx]
            return entry
        else:
            return self.data[i]



class PreTrainingDataset(torch.utils.data.Dataset):
    """Dataset for Pre-training. 
    """
    def __init__(self, data: List[Any], config: PreTrainingConfig, training: bool=True, mp_rank=0, mp_world_size=0):
        super().__init__()
        # if mp_world_size > 0:
        #     assert 0 <= mp_rank < mp_world_size
            
        #     text_paths = text_paths[_slice_chunk(mp_rank, mp_world_size, len(text_paths))]
        # logger.info(f"Totally {len(text_paths)} text files in the {mp_rank}-th process")
        
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
        return "\n".join(summary)
        
        
    def __getitem__(self, i):
        entry = self.data[i]
        
        if getattr(self.config, 'paired_task', 'None').lower() == 'nsp':
            paired_entry = random.choice(self.data)
        else:
            paired_entry = None
        
        example = self.config.exemplify(entry, paired_entry=paired_entry, training=self.training)
        return example
        
        
    def collate(self, batch_examples: List[str]):
        batch = self.config.batchify(batch_examples)
        return Batch(**batch)
