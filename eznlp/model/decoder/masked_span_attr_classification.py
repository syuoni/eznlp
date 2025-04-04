# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...nn.modules import CombinedDropout
from ...nn.init import reinit_embedding_, reinit_layer_
from ..encoder import EncoderConfig
from .base import SingleDecoderConfigBase, DecoderBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .span_attr_classification import ChunkSinglesDecoderMixin

logger = logging.getLogger(__name__)


class MaskedSpanAttrClsDecoderConfig(SingleDecoderConfigBase, ChunkSinglesDecoderMixin):
    def __init__(self, **kwargs):
        self.size_emb_dim = kwargs.pop('size_emb_dim', 0)
        self.label_emb_dim = kwargs.pop('label_emb_dim', 0)
        self.reduction = kwargs.pop('reduction', EncoderConfig(arch='FFN', hid_dim=150, num_layers=1, in_drop_rates=(0.0, 0.0, 0.0), hid_drop_rate=0.0))
        
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop('hid_drop_rates', (0.2, 0.0, 0.0))
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        
        self.ck_loss_weight = kwargs.pop('ck_loss_weight', 0)
        
        self.check_ac_labels = kwargs.pop('check_ac_labels', False)
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        
        # Change the default as multi-label classification
        kwargs['multilabel'] = kwargs.pop('multilabel', True)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self._name_sep.join([self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'hid_drop_rates', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def in_dim(self):
        return self._in_dim
        
    @in_dim.setter
    def in_dim(self, dim: int): 
        if dim is not None: 
            self._in_dim = dim
            self.reduction.in_dim = dim + self.size_emb_dim + self.label_emb_dim
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())
        
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        self.max_size_id = math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        
        counter = Counter(label for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        counter = Counter((label, chunk[0]) for data in partitions for entry in data for label, chunk in entry['attributes'])
        self.existing_ac_labels = set(list(counter.keys()))
        
        
    def exemplify(self, entry: dict, training: bool=True):
        example = super().exemplify(entry, training)
        
        # Attention mask for each example
        cs_obj = example['cs_obj']
        num_tokens, num_chunks = cs_obj.num_tokens, len(cs_obj.chunks)
        example['masked_span_bert_like'] = {'span_size_ids': cs_obj.span_size_ids}
        
        ck2tok_mask = torch.ones(num_chunks, num_tokens, dtype=torch.bool)
        for i, (label, start, end) in enumerate(cs_obj.chunks):
            for j in range(start, end):
                ck2tok_mask[i, j] = False
        example['masked_span_bert_like'].update({'ck2tok_mask': ck2tok_mask})
        
        return example
        
        
    def batchify(self, batch_examples: List[dict], batch_sub_mask: torch.Tensor):
        batch = super().batchify(batch_examples)
        
        # Batchify chunk-to-token attention mask
        # Remove `[CLS]` and `[SEP]` 
        batch_sub_mask = batch_sub_mask[:, 2:]
        batch['masked_span_bert_like'] = {}
        for tensor_name in batch_examples[0]['masked_span_bert_like'].keys(): 
            if tensor_name.endswith('_mask'): 
                max_num_items = max(ex['masked_span_bert_like'][tensor_name].size(0) for ex in batch_examples)
                batch_item2tok_mask = batch_sub_mask.unsqueeze(1).repeat(1, max_num_items, 1)
                
                for i, ex in enumerate(batch_examples): 
                    item2tok_mask = ex['masked_span_bert_like'][tensor_name]
                    num_items, num_tokens = item2tok_mask.size()
                    batch_item2tok_mask[i, :num_items, :num_tokens].logical_or_(item2tok_mask)
                
                batch['masked_span_bert_like'].update({tensor_name: batch_item2tok_mask})
                
            elif tensor_name.endswith('_ids'):
                batch_ids = [ex['masked_span_bert_like'][tensor_name] for ex in batch_examples]
                batch_ids = torch.nn.utils.rnn.pad_sequence(batch_ids, batch_first=True, padding_value=0)
                batch['masked_span_bert_like'].update({tensor_name: batch_ids})
        
        return batch
        
        
    def instantiate(self):
        return MaskedSpanAttrClsDecoder(self)



class MaskedSpanAttrClsDecoder(DecoderBase, ChunkSinglesDecoderMixin):
    def __init__(self, config: MaskedSpanAttrClsDecoderConfig):
        super().__init__()
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.multilabel = config.multilabel
        self.conf_thresh = config.conf_thresh
        self.ck_loss_weight = config.ck_loss_weight
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.check_ac_labels = config.check_ac_labels
        self.existing_ac_labels = config.existing_ac_labels
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.label_emb_dim)
            reinit_embedding_(self.label_embedding)
        
        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)
        
        if config.reduction.num_layers > 0 and config.reduction.out_dim > 0:
            self.reduction = config.reduction.instantiate()
            self.hid2logit = torch.nn.Linear(config.reduction.out_dim, config.voc_dim)
        else:
            self.hid2logit = torch.nn.Linear(config.reduction.in_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        if config.ck_loss_weight > 0: 
            self.ck_hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim+config.label_emb_dim, config.ck_voc_dim)
            reinit_layer_(self.ck_hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling. 
        """
        for cs_obj, chunks_pred in zip(batch.cs_objs, batch_chunks_pred):
            if cs_obj.chunks_pred is None:
                cs_obj.chunks_pred = chunks_pred
                cs_obj.build(self)
                cs_obj.to(self.hid2logit.weight.device)
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        # span_query_hidden: (batch, num_chunks, hid_dim)
        batch_logits = []
        for i, cs_obj in enumerate(batch.cs_objs):
            num_chunks = len(cs_obj.chunks)
            if num_chunks == 0:
                logits = None
            else: 
                # span_hidden: (num_chunks, hid_dim)
                span_hidden = span_query_hidden[i, :len(cs_obj.chunks)]
                
                if hasattr(self, 'size_embedding'):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cs_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
                
                if hasattr(self, 'label_embedding'):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cs_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, label_embedded], dim=-1)
                
                span_hidden = self.in_dropout(span_hidden)
                
                if hasattr(self, 'reduction'):
                    reduced = self.reduction(span_hidden)
                    logits = self.hid2logit(self.hid_dropout(reduced))
                else:
                    # logits: (num_chunks, logit_dim)
                    logits = self.hid2logit(span_hidden)
                
                if self.ck_loss_weight > 0:
                    ck_logits = self.ck_hid2logit(span_hidden)
                    logits = (logits, ck_logits)
            
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden, span_query_hidden)
        
        losses = []
        for logits, cs_obj in zip(batch_logits, batch.cs_objs):
            if len(cs_obj.chunks) == 0:
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cs_obj.cs2label_id
                ck_label_ids_gold = cs_obj.ck_label_ids_gold
                if self.ck_loss_weight > 0: 
                    logits, ck_logits = logits
                if hasattr(cs_obj, 'non_mask'):
                    non_mask = cs_obj.non_mask
                    logits, label_ids = logits[non_mask], label_ids[non_mask]
                loss = self.criterion(logits, label_ids)
                if self.ck_loss_weight > 0: 
                    loss = loss + self.ck_loss_weight*self.criterion(ck_logits, ck_label_ids_gold)
            losses.append(loss)
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, span_query_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden, span_query_hidden)
        
        batch_attributes = []
        for logits, cs_obj in zip(batch_logits, batch.cs_objs):
            if len(cs_obj.chunks) == 0:
                attributes = []
            else:
                if self.ck_loss_weight > 0:
                    logits, _ = logits
                if not self.multilabel:
                    confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
                    labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
                    attributes = [(label, chunk) for label, chunk in zip(labels, cs_obj.chunks) if label != self.none_label]
                    confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
                else:
                    all_confidences = logits.sigmoid()
                    # Zero-out all entities according to <none> labels
                    all_confidences[all_confidences[:,self.none_idx] > (1-self.conf_thresh)] = 0
                    # Zero-out <none> labels for all entities
                    all_confidences[:,self.none_idx] = 0
                    assert all_confidences.size(0) == len(cs_obj.chunks)
                    
                    all_confidences_list = all_confidences.cpu().tolist()
                    pos_entries = torch.nonzero(all_confidences > self.conf_thresh).cpu().tolist()
                    
                    attributes = [(self.idx2label[i], cs_obj.chunks[cidx]) for cidx, i in pos_entries]
                    confidences = [all_confidences_list[cidx][i] for cidx, i in pos_entries]
                assert len(confidences) == len(attributes)
                attributes = self._filter(attributes)
            batch_attributes.append(attributes)
        
        return batch_attributes
