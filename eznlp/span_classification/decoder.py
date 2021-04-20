# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import random
import logging
import torch

from ..data.wrapper import TensorWrapper, Batch
from ..nn.init import reinit_embedding_
from ..model.decoder import DecoderConfig, Decoder

logger = logging.getLogger(__name__)


class SpanClassificationDecoderConfig(DecoderConfig):
    def __init__(self, **kwargs):
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        self.num_neg_chunks = kwargs.pop('num_neg_chunks', 100)
        self.max_span_size = kwargs.pop('max_span_size', 10)
        
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        super().__init__(**kwargs)
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['agg_mode', 'in_dim', 'in_drop_rates']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def idx2label(self):
        return self._idx2label
    
    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        
    @property
    def full_in_dim(self):
        return self.in_dim + self.size_emb_dim
        
    @property
    def voc_dim(self):
        return len(self.label2idx)
        
    @property
    def none_idx(self):
        return self.label2idx[self.none_label]
        
    def build_vocab(self, *partitions):
        counter = Counter()
        size_counter = Counter()
        for data in partitions:
            for data_entry in data:
                counter.update([ck[0] for ck in data_entry['chunks']])
                size_counter.update([ck[2]-ck[1] for ck in data_entry['chunks']])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        num_spans = sum(size_counter.values())
        num_oov_spans = sum([num for size, num in size_counter.items() if size > self.max_span_size])
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
            
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return Spans(data_entry, self, training=training)
        
    def batchify(self, batch_spans_objs: list):
        return batch_spans_objs
        
    def instantiate(self):
        return SpanClassificationDecoder(self)



class Spans(TensorWrapper):
    """
    A wrapper of spans with original chunks. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': list}
    """
    def __init__(self, data_entry: dict, config: SpanClassificationDecoderConfig, training: bool=True):
        if 'chunks' in data_entry:
            self.chunks = data_entry['chunks']
            pos_spans = [(start, end) for label, start, end in data_entry['chunks']]
            pos_labels = [label for label, start, end in data_entry['chunks']]
        else:
            pos_spans, pos_labels = [], []
        
        num_tokens = len(data_entry['tokens'])
        neg_spans = []
        for start in range(num_tokens):
            for end in range(start+1, min(start+1+config.max_span_size, num_tokens+1)):
                if (start, end) not in pos_spans:
                    neg_spans.append((start, end))
                    
        if training and len(neg_spans) > config.num_neg_chunks:
            neg_spans = random.sample(neg_spans, config.num_neg_chunks)
        
        self.spans = pos_spans + neg_spans
        # Note: size_id = size - 1
        self.span_size_ids = torch.tensor([end-start-1 for start, end in self.spans])
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        
        if 'chunks' in data_entry:
            labels = pos_labels + [config.none_label] * len(neg_spans)
            self.label_ids = torch.tensor([config.label2idx[label] for label in labels])



class SpanClassificationDecoder(Decoder):
    def __init__(self, config: SpanClassificationDecoderConfig):
        super().__init__(config)
        
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.label2idx = config.label2idx
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        self.size_embedding = torch.nn.Embedding(config.max_span_size, config.size_emb_dim)
        reinit_embedding_(self.size_embedding)
        
        
    def get_span_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_span_logits = []
        for k in range(batch.size):
            # span_hidden: (num_spans, hid_dim)
            span_hidden = torch.stack([full_hidden[k, start:end].max(dim=0).values for start, end in batch.spans_objs[k].spans])
            # size_embedded: (num_spans, emb_dim)
            size_embedded = self.size_embedding(batch.spans_objs[k].span_size_ids)
            span_logits = self.hid2logit(self.dropout(torch.cat([span_hidden, size_embedded], dim=-1)))
            batch_span_logits.append(span_logits)
            
        return batch_span_logits
    
    
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_span_logits = self.get_span_logits(batch, full_hidden)
        
        losses = [self.criterion(batch_span_logits[k], batch.spans_objs[k].label_ids) for k in range(batch.size)]
        return torch.stack(losses)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_span_logits = self.get_span_logits(batch, full_hidden)
        
        batch_chunks = []
        for k in range(batch.size):
            labels = [self.idx2label[i] for i in batch_span_logits[k].argmax(dim=-1).cpu().tolist()]
            
            chunks = [(label, start, end) for label, (start, end) in zip(labels, batch.spans_objs[k].spans) if label != self.none_label]
            batch_chunks.append(chunks)
            
        return batch_chunks
    
    