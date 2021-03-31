# -*- coding: utf-8 -*-
from typing import List
import tqdm
import logging
import numpy
import transformers

from ..data.dataset import Dataset
from .classifier import TextClassifierConfig
from ..sequence_tagging.transition import find_ascending

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, data: List[dict], config: TextClassifierConfig=None):
        """
        Parameters
        ----------
        data : list of dict
            [{'tokens': TokenSequence, 'label': str/int}, ...]
            
            `label` is an optional key, which does not exist if the data are unlabeled. 
        """
        if config is None:
            config = TextClassifierConfig()
        super().__init__(data, config)
        
    @property
    def summary(self):
        summary = [super().summary]
        
        n_labels = len(self.config.decoder.label2idx)
        summary.append(f"The dataset has {n_labels:,} labels")
        return "\n".join(summary)
    
    
    
def truncate_for_bert_like(data: list, tokenizer: transformers.PreTrainedTokenizer, mode: str='head+tail', verbose=True):
    """
    Truncation methods:
        1. head-only: keep the first 510 tokens;
        2. tail-only: keep the last 510 tokens;
        3. head+tail: empirically select the first 128 and the last 382 tokens.
    
    References
    ----------
    [1] Sun et al. 2019. How to fine-tune BERT for text classification? CCL 2019. 
    """
    max_len = tokenizer.model_max_length - 2
    if mode.lower() == 'head-only':
        head_len, tail_len = max_len, 0
    elif mode.lower() == 'tail-only':
        head_len, tail_len = 0, max_len
    else:
        head_len = tokenizer.model_max_length // 4
        tail_len = max_len - head_len
        
    n_truncated = 0
    for data_entry in tqdm.tqdm(data, disable=not verbose, ncols=100, desc="Truncating data"):
        tokens = data_entry['tokens']
        nested_sub_tokens = [tokenizer.tokenize(word) for word in tokens.raw_text]
        sub_tok_seq_lens = [len(tok) for tok in nested_sub_tokens]
        
        if sum(sub_tok_seq_lens) > max_len:
            cum_lens = numpy.cumsum(sub_tok_seq_lens).tolist()
            
            # head_end/tail_begin will be 0 if head_len/tail_len == 0
            find_head, head_end = find_ascending(cum_lens, head_len)
            if find_head:
                head_end += 1
                
            rev_cum_lens = numpy.cumsum(sub_tok_seq_lens[::-1]).tolist()
            find_tail, tail_begin = find_ascending(rev_cum_lens, tail_len)
            if find_tail:
                tail_begin += 1
                
            if tail_len == 0:
                data_entry['tokens'] = tokens[:head_end]
            elif head_len == 0:
                data_entry['tokens'] = tokens[-tail_begin:]
            else:
                data_entry['tokens'] = tokens[:head_end] + tokens[-tail_begin:]
                
            n_truncated += 1
            
    logger.info(f"Truncated sequences: {n_truncated} ({n_truncated/len(data)*100:.2f}%)")
    return data
    
