# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence

from ..data import TensorWrapper, Batch
from ..data import Dataset
from .classifier import TextClassifierConfig


class TextClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: TextClassifierConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 
              'label': str/int}, ...]
            
            `label` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = TextClassifierConfig()
        super().__init__(data, config)
        
        self._is_labeled = ('label' in data[0])
        
        self._building_vocabs = (self.config.decoder.idx2label is None)
        if self._building_vocabs:
            assert self._is_labeled
            
            self._build_token_vocab()
            self._build_label_vocab()
            
            if self.config.embedder.char is not None:
                self._build_char_vocab()
            
            if self.config.embedder.enum is not None:
                self._build_enum_vocabs()
                
            self.config._update_dims(self.data[0]['tokens'][0])
        else:
            if self._is_labeled:
                self._check_label_vocab()
                
                
    def _build_label_vocab(self):
        counter = Counter([curr_data['label'] for curr_data in self.data])
        self.config.decoder.set_vocab(idx2label=list(counter.keys()))
        
    def _check_label_vocab(self):
        counter = Counter([curr_data['label'] for curr_data in self.data])
        
        oov = [label for label in counter if label not in self.config.decoder.label2idx]
        if len(oov) > 0:
            raise RuntimeError(f"OOV labels exist: {oov}")
            
    def extra_summary(self):
        n_labels = len(self.config.decoder.label2idx)
        return f"The dataset has {n_labels:,} labels"
        
        
    def __getitem__(self, i):
        curr_data = self.data[i]
        
        tok_ids = self.config.embedder.token.trans(curr_data['tokens'].text)
        
        if self.config.embedder.char is not None:
            char_ids = [self.config.embedder.char.trans(tok) for tok in curr_data['tokens'].raw_text]
        else:
            char_ids = None
        if self.config.embedder.enum is not None:
            enum_feats = {f: enum_config.trans(getattr(curr_data['tokens'], f)) for f, enum_config in self.config.embedder.enum.items()}
        else:
            enum_feats = None
        if self.config.embedder.val is not None:
            val_feats = {f: val_config.trans(getattr(curr_data['tokens'], f)) for f, val_config in self.config.embedder.val.items()}
        else:
            val_feats = None
            
        # Prepare text for pretrained embedders
        tokenized_raw_text = curr_data['tokens'].raw_text
        
        label_id = torch.tensor(self.config.decoder.label2idx[curr_data['label']]) if self._is_labeled else None
        return TensorWrapper(char_ids=char_ids, tok_ids=tok_ids, enum=enum_feats, val=val_feats, 
                             tokenized_raw_text=tokenized_raw_text, 
                             label_id=label_id)
    
    
    def collate(self, batch_examples):
        batch_tok_ids = [ex.tok_ids for ex in batch_examples]
        seq_lens = torch.tensor([seq.size(0) for seq in batch_tok_ids])
        batch_tok_ids = pad_sequence(batch_tok_ids, batch_first=True, padding_value=self.config.embedder.token.pad_idx)
        
        if self.config.embedder.char is not None:
            batch_char_ids = [tok for ex in batch_examples for tok in ex.char_ids]
            tok_lens = torch.tensor([tok.size(0) for tok in batch_char_ids])
            batch_char_ids = pad_sequence(batch_char_ids, batch_first=True, padding_value=self.config.embedder.char.pad_idx)
        else:
            batch_char_ids, tok_lens = None, None
            
        if self.config.embedder.enum is not None:
            batch_enum = {f: pad_sequence([ex.enum[f] for ex in batch_examples], batch_first=True, 
                                          padding_value=enum_config.pad_idx) for f, enum_config in self.config.embedder.enum.items()}
        else:
            batch_enum = None
            
        if self.config.embedder.val is not None:
            batch_val = {f: pad_sequence([ex.val[f] for ex in batch_examples], batch_first=True, 
                                         padding_value=0.0) for f in self.config.embedder.val.keys()}
        else:
            batch_val = None
        
        # Prepare text for pretrained embedders
        batch_tokenized_raw_text = [ex.tokenized_raw_text for ex in batch_examples]
        
        batch_label_id = torch.stack([ex.label_id for ex in batch_examples]) if self._is_labeled else None
        batch = Batch(tok_ids=batch_tok_ids, seq_lens=seq_lens, 
                      char_ids=batch_char_ids, tok_lens=tok_lens, 
                      enum=batch_enum, val=batch_val, 
                      tokenized_raw_text=batch_tokenized_raw_text, 
                      label_id=batch_label_id)
        
        batch.build_masks({'tok_mask': (seq_lens, batch_tok_ids.size(1))})
        if self.config.embedder.char is not None:
            batch.build_masks({'char_mask': (tok_lens, batch_char_ids.size(1))})
        return batch
    
    