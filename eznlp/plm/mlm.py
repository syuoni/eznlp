# -*- coding: utf-8 -*-
from typing import List
import random
import torch
import transformers

from ..nn.functional import seq_lens2mask
from .base import PreTrainingConfig



class MaskedLMConfig(PreTrainingConfig):
    """Configurations for masked LM pretraining, optionally with a sentence pair task (e.g., NSP, SOP). 
    
    """
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop('bert_like')
        
        # Masked LM 
        self.masking_rate = kwargs.pop('masking_rate', 0.15)
        self.masking_rate_dev = kwargs.pop('masking_rate_dev', 0.0)
        self.random_word_rate = kwargs.pop('random_word_rate', 0.1)
        self.unchange_rate = kwargs.pop('unchange_rate', 0.1)
        self.use_wwm = kwargs.pop('use_wwm', False)
        self.ngram_weights = kwargs.pop('ngram_weights', (1.0, ))
        
        # Sentence pair task: None/NSP/SOP
        self.paired_task = kwargs.pop('paired_task', 'None')
        if self.paired_task.lower() == 'none':
            assert isinstance(self.bert_like, transformers.BertForMaskedLM)
        else:
            assert isinstance(self.bert_like, transformers.BertForPreTraining)
        
        super().__init__(**kwargs)
        
        
    @property
    def mlm_label_mask_id(self):
        # transformers/models/bert/modeling_bert.py/BertForPreTraining
        # Tokens with indices set to `-100` are ignored (masked), the loss is only 
        # computed for the tokens with labels in `[0, ..., config.vocab_size]`
        return -100
        
    @property
    def sentence_A_id(self):
        # token_type_ids: 
        # - 0 corresponds to a `sentence A` token,
        # - 1 corresponds to a `sentence B` token.
        return 0 
        
    @property
    def sentence_B_id(self):
        return 1
        
        
    def dynamic_mask_for_lm(self, entry: dict):
        """Convert an `entry` to `mlm_tok_ids` and `mlm_lab_ids`, with dynamic masking. 
        
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
        
        # (1) Select which positions are masked
        ## (1.1) Prepare probabilities
        masking_rate = random.uniform(self.masking_rate-self.masking_rate_dev, self.masking_rate+self.masking_rate_dev)
        ngram_weights = [w/n for n, w in enumerate(self.ngram_weights, 1)]
        masking_rate = masking_rate * sum(ngram_weights)  # Adjust masking rate 
        ngram_weights = [w/sum(ngram_weights) for w in ngram_weights]  # Re-normalize N-gram weights
        
        ## (1.2) Select span ids
        mask_span_ids = random.sample(range(num_spans), max(int(num_spans*masking_rate+0.5), 1))  # Sampling without replacement
        ngrams = random.choices(range(1, len(ngram_weights)+1), weights=ngram_weights, k=len(mask_span_ids))  # Multinomial sampling
        mask_span_ids = list(set([min(mask_span_ids[k]+i, num_spans-1) for k, n in enumerate(ngrams) for i in range(n)]))
        
        ## (1.3) Unfold span ids to position ids
        mask_pos_ids  = [i for span_id in mask_span_ids for i in range(*wwm_spans[span_id])]
        
        # (2) Decide which positions are replaced with `[MASK]`, random token, or unchanged, by multinomial sampling
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
        
        return mlm_tok_ids, mlm_lab_ids
        
        
    def exemplify(self, entry: dict, paired_entry: dict=None, training: bool=True):
        """Use dynamic masking. 
        
        entry: dict / str / List[str]
            {'rejoined_text': str, 'wwm_cuts': List[int], ...}
        """
        mlm_tok_ids, mlm_lab_ids = self.dynamic_mask_for_lm(entry)
        
        if self.paired_task.lower() == 'nsp':
            # Next sentence prediction
            max_len = self.tokenizer.model_max_length - 3
            len1 = min(len(mlm_tok_ids), max_len)
            new_len1 = int(random.uniform(0.25, 0.75)*len1 + 0.5)
            
            if random.random() < 0.5:
                # 0 indicates sequence B is a continuation of sequence A
                mlm_tok_ids = mlm_tok_ids[:new_len1] + [self.sep_id] + mlm_tok_ids[new_len1:len1]
                mlm_lab_ids = mlm_lab_ids[:new_len1] + [self.mlm_label_mask_id] + mlm_lab_ids[new_len1:len1]
                tok_type_ids = [self.sentence_A_id] * (new_len1+1) + [self.sentence_B_id] * (len(mlm_tok_ids)-new_len1-1)
                paired_lab_id = 0
                
            else: 
                # 1 indicates sequence B is a random sequence
                paired_mlm_tok_ids, paired_mlm_lab_ids = self.dynamic_mask_for_lm(paired_entry)
                len2 = len(paired_mlm_tok_ids)
                new_len2 = min(len1-new_len1, len2)
                
                mlm_tok_ids = mlm_tok_ids[:new_len1] + [self.sep_id] + paired_mlm_tok_ids[(len2-new_len2):]
                mlm_lab_ids = mlm_lab_ids[:new_len1] + [self.mlm_label_mask_id] + paired_mlm_lab_ids[(len2-new_len2):]
                tok_type_ids = [self.sentence_A_id] * (new_len1+1) + [self.sentence_B_id] * (len(mlm_tok_ids)-new_len1-1)
                paired_lab_id = 1
            
        elif self.paired_task.lower() == 'sop':
            # Sentence order prediction
            max_len = self.tokenizer.model_max_length - 3
            len1 = min(len(mlm_tok_ids), max_len)
            new_len1 = int(random.uniform(0.25, 0.75)*len1 + 0.5)
            
            if random.random() < 0.5:
                # 0 indicates sequence B is a continuation of sequence A
                mlm_tok_ids = mlm_tok_ids[:new_len1] + [self.sep_id] + mlm_tok_ids[new_len1:len1]
                mlm_lab_ids = mlm_lab_ids[:new_len1] + [self.mlm_label_mask_id] + mlm_lab_ids[new_len1:len1]
                tok_type_ids = [self.sentence_A_id] * (new_len1+1) + [self.sentence_B_id] * (len(mlm_tok_ids)-new_len1-1)
                paired_lab_id = 0
                
            else:
                # 1 indicates sequence A is a continuation of sequence B
                mlm_tok_ids = mlm_tok_ids[new_len1:len1] + [self.sep_id] + mlm_tok_ids[:new_len1]
                mlm_lab_ids = mlm_lab_ids[new_len1:len1] + [self.mlm_label_mask_id] + mlm_lab_ids[:new_len1]
                tok_type_ids = [self.sentence_A_id] * (len(mlm_tok_ids)-new_len1) + [self.sentence_B_id] * new_len1
                paired_lab_id = 1
        
        # Add `[CLS]` and `[SEP]`
        mlm_tok_ids = [self.cls_id] + mlm_tok_ids + [self.sep_id]
        mlm_lab_ids = [self.mlm_label_mask_id] + mlm_lab_ids + [self.mlm_label_mask_id]
        example = {'mlm_tok_ids': torch.tensor(mlm_tok_ids), 
                   'mlm_lab_ids': torch.tensor(mlm_lab_ids)}
        
        if self.paired_task.lower() != 'none':
            tok_type_ids = [self.sentence_A_id] + tok_type_ids + [self.sentence_B_id]
            example.update({'tok_type_ids': torch.tensor(tok_type_ids), 
                            'paired_lab_id': torch.tensor(paired_lab_id)})
        
        return example
        
        
    def batchify(self, batch_ex: List[dict]):
        batch_mlm_tok_ids = [ex['mlm_tok_ids'] for ex in batch_ex]
        batch_mlm_lab_ids = [ex['mlm_lab_ids'] for ex in batch_ex]
        
        mlm_tok_seq_lens = torch.tensor([s.size(0) for s in batch_mlm_tok_ids])
        mlm_att_mask = seq_lens2mask(mlm_tok_seq_lens)
        batch_mlm_tok_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_tok_ids, batch_first=True, padding_value=self.pad_id)
        batch_mlm_lab_ids = torch.nn.utils.rnn.pad_sequence(batch_mlm_lab_ids, batch_first=True, padding_value=self.mlm_label_mask_id)
        
        batch = {'mlm_tok_ids': batch_mlm_tok_ids, 
                 'mlm_lab_ids': batch_mlm_lab_ids, 
                 'mlm_att_mask': mlm_att_mask}
        
        if self.paired_task.lower() != 'none':
            batch_tok_type_ids = [ex['tok_type_ids'] for ex in batch_ex]
            batch_tok_type_ids = torch.nn.utils.rnn.pad_sequence(batch_tok_type_ids, batch_first=True, padding_value=self.sentence_A_id)
            batch_paired_lab_ids = torch.stack([ex['paired_lab_id'] for ex in batch_ex])
            batch.update({'tok_type_ids': batch_tok_type_ids, 
                          'paired_lab_ids': batch_paired_lab_ids})
        
        return batch
        
        
    def instantiate(self):
        return self.bert_like
