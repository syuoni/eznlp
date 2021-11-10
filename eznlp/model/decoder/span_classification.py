# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import random
import logging
import torch

from ...wrapper import TargetWrapper, Batch
from ...utils.chunk import detect_nested, filter_clashed_by_priority
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase

logger = logging.getLogger(__name__)


class SpanClassificationDecoderMixin(DecoderMixinBase):
    @property
    def idx2label(self):
        return self._idx2label
        
    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        
    @property
    def voc_dim(self):
        return len(self.label2idx)
        
    @property
    def none_idx(self):
        return self.label2idx[self.none_label]
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return {'spans_obj': Spans(data_entry, self, training=training)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'spans_objs': [ex['spans_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [spans_obj.chunks for spans_obj in batch.spans_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        """Micro-F1 for entity recognition. 
        
        References
        ----------
        https://www.clips.uantwerpen.be/conll2000/chunking/output.html
        """
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']



class Spans(TargetWrapper):
    """A wrapper of spans with underlying chunks. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple]}
    """
    def __init__(self, data_entry: dict, config: SpanClassificationDecoderMixin, training: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry.get('chunks', None)
        if training:
            pos_spans = [(start, end) for label, start, end in self.chunks]
        else:
            pos_spans = []
        
        neg_spans = [(start, end) 
                         for start in range(len(data_entry['tokens'])) 
                         for end in range(start+1, min(start+1+config.max_span_size, len(data_entry['tokens']) + 1))
                         if (start, end) not in pos_spans]
        
        if training and len(neg_spans) > config.num_neg_chunks:
            neg_spans = random.sample(neg_spans, config.num_neg_chunks)
        
        self.spans = pos_spans + neg_spans
        # Note: size_id = size - 1
        self.span_size_ids = torch.tensor([end-start-1 for start, end in self.spans], dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        
        # TODO: boundary/label smoothing
        # Do not smooth to `<none>` label
        if self.chunks is not None:
            span2label = {(start, end): label for label, start, end in self.chunks}
            labels = [span2label.get(span, config.none_label) for span in self.spans]
            self.label_ids = torch.tensor([config.label2idx[label] for label in labels], dtype=torch.long)



class SpanClassificationDecoderConfig(SingleDecoderConfigBase, SpanClassificationDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.num_neg_chunks = kwargs.pop('num_neg_chunks', 100)
        self.max_span_size = kwargs.pop('max_span_size', 10)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        # Note: non-nested overlapping chunks are never allowed
        self.allow_nested = kwargs.pop('allow_nested', None)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'agg_mode', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        self.allow_nested = any(detect_nested(entry['chunks']) for data in partitions for entry in data)
        if self.allow_nested:
            logger.info("Nested chunks detected, nested chunks are allowed in decoding...")
        else:
            logger.info("No nested chunks detected, only flat chunks are allowed in decoding...")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
        
        
    def instantiate(self):
        return SpanClassificationDecoder(self)



class SpanClassificationDecoder(DecoderBase, SpanClassificationDecoderMixin):
    def __init__(self, config: SpanClassificationDecoderConfig):
        super().__init__()
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.allow_nested = config.allow_nested
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_span_size, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for k in range(full_hidden.size(0)):
            # span_hidden: (num_spans, span_size, hid_dim) -> (num_spans, hid_dim)
            span_hidden = [full_hidden[k, start:end] for start, end in batch.spans_objs[k].spans]
            span_mask = seq_lens2mask(torch.tensor([h.size(0) for h in span_hidden], dtype=torch.long, device=full_hidden.device))
            span_hidden = torch.nn.utils.rnn.pad_sequence(span_hidden, batch_first=True, padding_value=0.0)
            span_hidden = self.aggregating(self.dropout(span_hidden), mask=span_mask)
            
            if hasattr(self, 'size_embedding'):
                # size_embedded: (num_spans, emb_dim)
                size_embedded = self.size_embedding(batch.spans_objs[k].span_size_ids)
                span_hidden = torch.cat([span_hidden, self.dropout(size_embedded)], dim=-1)
                
            logits = self.hid2logit(span_hidden)
            batch_logits.append(logits)
            
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        losses = [self.criterion(batch_logits[k], batch.spans_objs[k].label_ids) for k in range(full_hidden.size(0))]
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_chunks = []
        for k in range(full_hidden.size(0)):
            confidences, label_ids = batch_logits[k].softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
            chunks = [(label, start, end) for label, (start, end) in zip(labels, batch.spans_objs[k].spans) if label != self.none_label]
            confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            assert len(confidences) == len(chunks)
            
            # Sort chunks from high to low confidences
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_nested=self.allow_nested)
            
            batch_chunks.append(chunks)
        return batch_chunks
