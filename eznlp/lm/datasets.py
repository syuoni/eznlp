# -*- coding: utf-8 -*-
import string
import json
import random
from collections import Counter, OrderedDict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.experimental.vocab import Vocab
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor

from ..datasets_utils import TensorWrapper, Batch


class COVID19MLMDataset(Dataset):
    """
    Masked Language Modeling Dataset. 
    """
    def __init__(self, data, bert_tokenizer):
        super().__init__()
        self.data = data
        self.bert_tokenizer = bert_tokenizer
        
        self.wp2idx = bert_tokenizer.get_vocab()
        self.idx2wp = {idx: wp for wp, idx in self.wp2idx.items()}
        assert min(self.wp2idx.values()) == 0
        assert max(self.wp2idx.values()) == len(self.wp2idx) - 1
        
        # Tokens with indices set to ``-100`` are ignored (masked), the loss is only 
        # computed for the tokens with labels in ``[0, ..., config.vocab_size]
        self.MLM_label_mask_id = -100
        
        
    def summary(self):
        n_seqs = len(self.data)
        n_raws = len({curr_data['raw_idx'] for curr_data in self.data})
        max_len = max([len(curr_data['tokens']) for curr_data in self.data])
        print(f"The dataset consists {n_seqs} sequences built from {n_raws} raw entries")
        print(f"The max sequence length is {max_len}")
        
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i):
        """
        Dynamic Masking.
        """
        tokens = self.data[i]['tokens']
        tokens.build_word_pieces(self.bert_tokenizer)
        
        MLM_wp_ids = [self.bert_tokenizer.cls_token_id] \
                   + [self.wp2idx[wp] for wp in tokens.word_pieces] \
                   + [self.bert_tokenizer.sep_token_id]
        MLM_lb_ids = []
        for k, wp_id in enumerate(MLM_wp_ids):
            if (wp_id not in self.bert_tokenizer.all_special_ids) and (random.random() < 0.15):
                random_v = random.random()
                if random_v < 0.8:
                    # Replace with `[MASK]`
                    MLM_wp_ids[k] = self.bert_tokenizer.mask_token_id
                elif random_v < 0.9:
                    # Replace with a random word-piece
                    MLM_wp_ids[k] = random.randint(0, len(self.idx2wp)-1)
                else:
                    # Retain the original word-piece
                    pass
                MLM_lb_ids.append(wp_id)
            else:
                MLM_lb_ids.append(self.MLM_label_mask_id)
                
        return TensorWrapper(MLM_wp_ids=torch.tensor(MLM_wp_ids), 
                             MLM_lb_ids=torch.tensor(MLM_lb_ids))
    
    
    def collate(self, batch_examples):
        batch_MLM_wp_ids = []
        batch_MLM_lb_ids = []
        
        for ex in batch_examples:
            batch_MLM_wp_ids.append(ex.MLM_wp_ids)
            batch_MLM_lb_ids.append(ex.MLM_lb_ids)
        
        seq_lens = torch.tensor([s.size(0) for s in batch_MLM_wp_ids])
        batch_MLM_wp_ids = pad_sequence(batch_MLM_wp_ids, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)
        batch_MLM_lb_ids = pad_sequence(batch_MLM_lb_ids, batch_first=True, padding_value=self.MLM_label_mask_id)
        
        batch = Batch(seq_lens=seq_lens, MLM_wp_ids=batch_MLM_wp_ids, MLM_lb_ids=batch_MLM_lb_ids)
        batch.build_masks({'attention_mask': (batch_MLM_wp_ids.size(), seq_lens)})
        return batch
        
    
# class PMCMLMDataset(Dataset):
#     def __
    