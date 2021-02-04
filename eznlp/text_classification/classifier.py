# -*- coding: utf-8 -*-
from typing import List
import torch
import transformers

from ..data.wrapper import Batch
from ..model.model import ModelConfig, Model
from .decoder import TextClassificationDecoderConfig


class TextClassifierConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.decoder: TextClassificationDecoderConfig = kwargs.pop('decoder', TextClassificationDecoderConfig())
        super().__init__(**kwargs)
        
    @property
    def valid(self):
        return super().valid and (self.decoder is not None) and self.decoder.valid
        
    @property
    def name(self):
        extra_name = self.decoder.attention_scoring if self.decoder.use_attention else self.decoder.pooling_mode
        return "-".join([super().name, extra_name])
    
    def build_vocabs_and_dims(self, *partitions):
        super().build_vocabs_and_dims(*partitions)
        
        if self.intermediate is not None:
            self.decoder.in_dim = self.intermediate.out_dim
        else:
            self.decoder.in_dim = self.full_hid_dim
            
        self.decoder.build_vocab(*partitions)
        
    
    def exemplify(self, data_entry: dict):
        example = super().exemplify(data_entry['tokens'])
        if 'label' in data_entry:
            example['label_id'] = self.decoder.exemplify(data_entry)
        return example
        
    
    def batchify(self, batch_examples: List[dict]):
        batch = super().batchify(batch_examples)
        if 'label_id' in batch_examples[0]:
            batch['label_ids'] = self.decoder.batchify([ex['label_id'] for ex in batch_examples])
        return batch
    
    
    def instantiate(self):
        # Only check validity at the most outside level
        assert self.valid
        return TextClassifier(self)
    
    
class TextClassifier(Model):
    def __init__(self, config: TextClassifierConfig):
        super().__init__(config)
        self.decoder = config.decoder.instantiate()
        
    def forward(self, batch: Batch, return_hidden: bool=False):
        full_hidden = self.get_full_hidden(batch)
        losses = self.decoder(batch, full_hidden)
        
        # Return `hidden` for the `decode` method, to avoid duplicated computation. 
        if return_hidden:
            return losses, full_hidden
        else:
            return losses
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor=None):
        if full_hidden is None:
            full_hidden = self.get_full_hidden(batch)
            
        return self.decoder.decode(batch, full_hidden)
    
    
    
        
class BERTTextClassifier(Model):
    def __init__(self, config: TextClassifierConfig, bert_like: transformers.PreTrainedModel=None):
        super().__init__(config, bert_like=bert_like)
        
    def get_full_hidden(self, batch: Batch):
        full_hidden, (seq_lens, mask) = self.bert_like_embedder(batch)
        # Replace seq_lens / mask
        batch.seq_lens = seq_lens
        batch.tok_mask = mask
        
        if not hasattr(self, 'intermediate'):
            return full_hidden
        else:
            return self.intermediate(batch, full_hidden)
        
        