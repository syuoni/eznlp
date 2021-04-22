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


class RelationClassificationDecoderConfig(DecoderConfig):
    def __init__(self, **kwargs):
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        self.num_neg_relations = kwargs.pop('num_neg_relations', 100)
        self.max_span_size = kwargs.pop('max_span_size', 10)
        
        self.ck_size_emb_dim = kwargs.pop('ck_size_emb_dim', 25)
        self.ck_label_emb_dim = kwargs.pop('ck_label_emb_dim', 25)
        
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        self.rel_none_label = kwargs.pop('rel_none_label', '<none>')
        self.idx2rel_label = kwargs.pop('idx2rel_label', None)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self.agg_mode
    
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['agg_mode', 'in_dim', 'in_drop_rates']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def idx2ck_label(self):
        return self._idx2ck_label
    
    @idx2ck_label.setter
    def idx2ck_label(self, idx2ck_label: List[str]):
        self._idx2ck_label = idx2ck_label
        self.ck_label2idx = {l: i for i, l in enumerate(idx2ck_label)} if idx2ck_label is not None else None
        
    @property
    def idx2rel_label(self):
        return self._idx2rel_label
        
    @idx2rel_label.setter
    def idx2rel_label(self, idx2rel_label: List[str]):
        self._idx2rel_label = idx2rel_label
        self.rel_label2idx = {l: i for i, l in enumerate(idx2rel_label)} if idx2rel_label is not None else None
        
    @property
    def full_in_dim(self):
        return self.in_dim*3 + self.ck_size_emb_dim*2 + self.ck_label_emb_dim*2
        
    @property
    def ck_voc_dim(self):
        return len(self.ck_label2idx)
    
    @property
    def ck_none_idx(self):
        return self.ck_label2idx[self.ck_none_label]
        
    @property
    def rel_voc_dim(self):
        return len(self.rel_label2idx)
        
    @property
    def rel_none_idx(self):
        return self.rel_label2idx[self.rel_none_label]
    
    @property
    def voc_dim(self):
        return self.rel_voc_dim
        
    def build_vocab(self, *partitions):
        ck_counter = Counter()
        rel_counter = Counter()
        for data in partitions:
            for data_entry in data:
                ck_counter.update([ck[0] for ck in data_entry['chunks']])
                rel_counter.update([rel[0] for rel in data_entry['relations']])
        self.idx2ck_label = [self.ck_none_label] + list(ck_counter.keys())
        self.idx2rel_label = [self.rel_none_label] + list(rel_counter.keys())
        
    def exemplify(self, data_entry: dict, training: bool=True):
        return SpanPairs(data_entry, self, training=training)
        
    def batchify(self, batch_span_pairs_objs: list):
        return batch_span_pairs_objs
        
    def instantiate(self):
        return RelationClassificationDecoder(self)



class SpanPairs(TensorWrapper):
    """
    A wrapper of span-pairs with original relations. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': list, 
         'relations': list}
    """
    def __init__(self, data_entry: dict, config: RelationClassificationDecoderConfig, training: bool=True):
        if 'relations' in data_entry:
            self.relations = data_entry['relations']
            pos_sp_pairs = [(head[1], head[2], tail[1], tail[2]) for label, head, tail in data_entry['relations']]
            pos_ck_labels = [(head[0], tail[0]) for label, head, tail in data_entry['relations']]
            pos_rel_labels = [label for label, head, tail in data_entry['relations']]
        else:
            pos_sp_pairs, pos_ck_labels, pos_rel_labels = [], [], []
        
        neg_sp_pairs, neg_ck_labels = [], []
        for head in data_entry['chunks']:
            for tail in data_entry['chunks']:
                if head != tail and (head[1], head[2], tail[1], tail[2]) not in pos_sp_pairs:
                    neg_sp_pairs.append((head[1], head[2], tail[1], tail[2]))
                    neg_ck_labels.append((head[0], tail[0]))
                    
        if training and len(neg_sp_pairs) > config.num_neg_relations:
            sampling_indexes = random.sample(range(len(neg_sp_pairs)), config.num_neg_relations)
            neg_sp_pairs = [neg_sp_pairs[i] for i in sampling_indexes]
            neg_ck_labels = [neg_ck_labels[i] for i in sampling_indexes]
            
        self.sp_pairs = pos_sp_pairs + neg_sp_pairs
        # sp_pair_size_ids: (num_pairs, 2)
        # Note: size_id = size - 1
        self.sp_pair_size_ids = torch.tensor([[h_end-h_start-1, t_end-t_start-1] for h_start, h_end, t_start, t_end in self.sp_pairs])
        self.sp_pair_size_ids.masked_fill_(self.sp_pair_size_ids>=config.max_span_size, config.max_span_size-1)
        
        # ck_label_ids: (num_pairs, 2)
        self.ck_labels = pos_ck_labels + neg_ck_labels
        self.ck_label_ids = torch.tensor([[config.ck_label2idx[hckl], config.ck_label2idx[tckl]] for hckl, tckl in self.ck_labels])
        
        if 'relations' in data_entry:
            rel_labels = pos_rel_labels + [config.rel_none_label] * len(neg_sp_pairs)
            self.rel_label_ids = torch.tensor([config.rel_label2idx[label] for label in rel_labels])



class RelationClassificationDecoder(Decoder):
    def __init__(self, config: RelationClassificationDecoderConfig):
        super().__init__(config)
        
        self.rel_none_label = config.rel_none_label
        self.idx2rel_label = config.idx2rel_label
        self.rel_label2idx = config.rel_label2idx
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
        self.ck_size_embedding = torch.nn.Embedding(config.max_span_size, config.ck_size_emb_dim)
        reinit_embedding_(self.ck_size_embedding)
        self.ck_label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.ck_label_emb_dim)
        reinit_embedding_(self.ck_label_embedding)
        # Trainable context vector for overlapping chunks
        self.zero_context = torch.nn.Parameter(torch.zeros(config.in_dim))
        
        
    def get_span_pair_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_span_pair_logits = []
        for k in range(batch.size):
            # span_pair_hidden: (num_pairs, hid_dim)
            
            head_hidden, tail_hidden, contexts = [], [], []
            for h_start, h_end, t_start, t_end in batch.span_pairs_objs[k].sp_pairs:
                head_hidden.append(full_hidden[k, h_start:h_end].max(dim=0).values)
                tail_hidden.append(full_hidden[k, t_start:t_end].max(dim=0).values)
                
                if h_end < t_start:
                    context = full_hidden[k, h_end:t_start].max(dim=0).values
                elif t_end < h_start:
                    context = full_hidden[k, t_end:h_start].max(dim=0).values
                else:
                    context = self.zero_context
                contexts.append(context)
            
            # head_hiddem/tail_hidden/contexts: (num_pairs, hid_dim)
            head_hidden = torch.stack(head_hidden)
            tail_hidden = torch.stack(tail_hidden)
            contexts = torch.stack(contexts)
            
            # ck_size_embedded/ck_label_embedded: (num_pairs, 2, emb_dim) -> (num_pairs, emb_dim*2)
            ck_size_embedded = self.ck_size_embedding(batch.span_pairs_objs[k].sp_pair_size_ids).flatten(start_dim=1)
            ck_label_embedded = self.ck_label_embedding(batch.span_pairs_objs[k].ck_label_ids).flatten(start_dim=1)
            
            span_pair_logits = self.hid2logit(self.dropout(torch.cat([head_hidden, tail_hidden, contexts, ck_size_embedded, ck_label_embedded], dim=-1)))
            batch_span_pair_logits.append(span_pair_logits)
            
        return batch_span_pair_logits
    
    
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_span_pair_logits = self.get_span_pair_logits(batch, full_hidden)
        
        losses = [self.criterion(batch_span_pair_logits[k], batch.span_pairs_objs[k].rel_label_ids) for k in range(batch.size)]
        return torch.stack(losses)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_span_pair_logits = self.get_span_pair_logits(batch, full_hidden)
        
        batch_relations = []
        for k in range(batch.size):
            rel_labels = [self.idx2rel_label[i] for i in batch_span_pair_logits[k].argmax(dim=-1).cpu().tolist()]
            
            relations = [(rel_label, (hckl, h_start, h_end), (tckl, t_start, t_end)) \
                             for rel_label, (h_start, h_end, t_start, t_end), (hckl, tckl) \
                             in zip(rel_labels, batch.span_pairs_objs[k].sp_pairs, batch.span_pairs_objs[k].ck_labels) \
                             if rel_label != self.rel_none_label]
            batch_relations.append(relations)
            
        return batch_relations
    
    