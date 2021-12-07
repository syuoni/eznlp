# -*- coding: utf-8 -*-
from typing import List
import random
import torch
import transformers

from ..nn.functional import seq_lens2mask
from ..config import Config


class PreTrainingConfig(Config):
    def __init__(self, **kwargs):
        self.tokenizer: transformers.PreTrainedTokenizer = kwargs.pop('tokenizer')
        self.stoi = self.tokenizer.get_vocab()
        self.special_ids = list(set(self.tokenizer.all_special_ids))
        self.non_special_ids = [idx for idx in self.stoi.values() if idx not in self.special_ids]
        
        super().__init__(**kwargs)
        
    @property
    def cls_id(self):
        return self.tokenizer.cls_token_id
        
    @property
    def sep_id(self):
        return self.tokenizer.sep_token_id
        
    @property
    def unk_id(self):
        return self.tokenizer.unk_token_id
        
    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id
        
    @property
    def mask_id(self):
        return self.tokenizer.mask_token_id



class MaskedLMConfig(PreTrainingConfig):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        
        self.use_wwm = kwargs.pop('use_wwm', False)
        self.ngram_weights = kwargs.pop('ngram_weights', (1.0, ))
        
        self.masking_rate = kwargs.pop('masking_rate', 0.15)
        self.masking_rate_dev = kwargs.pop('masking_rate_dev', 0.0)
        self.random_word_rate = kwargs.pop('random_word_rate', 0.1)
        self.unchange_rate = kwargs.pop('unchange_rate', 0.1)
        
        super().__init__(**kwargs)
        
        
    @property
    def mlm_label_mask_id(self):
        # transformers/models/bert/modeling_bert.py/BertForPreTraining
        # Tokens with indices set to `-100` are ignored (masked), the loss is only 
        # computed for the tokens with labels in `[0, ..., config.vocab_size]`
        return -100
        
        
    def exemplify(self, entry: dict, training: bool=True):
        """Use dynamic masking. 
        
        entry: dict / str / List[str]
            {'rejoined_text': str, 'wwm_cuts': List[int], ...}
        """
        tokenized_text = entry['rejoined_text'].split(" ")
        
        if self.use_wwm:
            wwm_cuts = entry['wwm_cuts']
            wwm_spans = [(start, end) for start, end in zip(wwm_cuts[:-1], wwm_cuts[1:])]
        else:
            wwm_spans = [(k, k+1) for k in range(len(tokenized_text))]
        
        mlm_tok_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        mlm_lab_ids = [self.mlm_label_mask_id] * len(mlm_tok_ids)
        
        num_spans = len(wwm_spans)
        num_tokens = len(mlm_tok_ids)
        
        # Select which positions are masked
        ## (1) Prepare probabilities
        masking_rate = random.uniform(self.masking_rate-self.masking_rate_dev, self.masking_rate+self.masking_rate_dev)
        ngram_weights = [w/n for n, w in enumerate(self.ngram_weights, 1)]
        masking_rate = masking_rate * sum(ngram_weights)  # Adjust masking rate 
        ngram_weights = [w/sum(ngram_weights) for w in ngram_weights]  # Re-normalize N-gram weights
        
        ## (2) Select span ids
        mask_span_ids = random.sample(range(num_spans), max(int(num_spans*masking_rate+0.5), 1))  # Without replacement
        ngrams = random.choices(range(1, len(ngram_weights)+1), weights=ngram_weights, k=len(mask_span_ids))  # Multinomial sampling
        mask_span_ids = list(set([min(mask_span_ids[k]+i, num_spans-1) for k, n in enumerate(ngrams) for i in range(n)]))
        
        ## (3) Unfold span ids to position ids
        mask_pos_ids  = [i for span_id in mask_span_ids for i in range(*wwm_spans[span_id])]
        
        # Decide which positions are replaced with `[MASK]`, random token, or unchanged
        operations = random.choices(range(3), k=len(mask_pos_ids), 
                                    weights=(1-self.random_word_rate-self.unchange_rate, self.random_word_rate, self.unchange_rate))
        
        for i, op in zip(mask_pos_ids, operations):
            mlm_lab_ids[i] = mlm_tok_ids[i]
            if op == 0:  # Replace with `[MASK]`
                mlm_tok_ids[i] = self.mask_id
            elif op == 1:  # Replace with a random token
                mlm_tok_ids[i] = random.choice(self.non_special_ids)
            else:  # Keep the original token unchange
                pass
        
        # Add `[CLS]` and `[SEP]`
        mlm_tok_ids = [self.cls_id] + mlm_tok_ids + [self.sep_id]
        mlm_lab_ids = [self.mlm_label_mask_id] + mlm_lab_ids + [self.mlm_label_mask_id]
        
        return {'mlm_tok_ids': torch.tensor(mlm_tok_ids), 
                'mlm_lab_ids': torch.tensor(mlm_lab_ids)}
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_mlm_tok_ids = [ex['mlm_tok_ids'] for ex in batch_ex]
        batch_mlm_lab_ids = [ex['mlm_lab_ids'] for ex in batch_ex]
        
        mlm_tok_seq_lens = torch.tensor([s.size(0) for s in batch_mlm_tok_ids])
        mlm_att_mask = seq_lens2mask(mlm_tok_seq_lens)
        batch_mlm_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_tok_ids, batch_first=True, padding_value=self.pad_id)
        batch_mlm_lab_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_lab_ids, batch_first=True, padding_value=self.mlm_label_mask_id)
        
        return {'mlm_tok_ids': batch_mlm_tok_ids, 
                'mlm_lab_ids': batch_mlm_lab_ids, 
                'mlm_att_mask': mlm_att_mask}
        
        
    def instantiate(self):
        return self.bert_like
