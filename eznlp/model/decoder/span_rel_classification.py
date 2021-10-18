# -*- coding: utf-8 -*-
from typing import List
from collections import Counter
import random
import logging
import torch

from ...wrapper import TargetWrapper, Batch
from ...utils.chunk import chunk_pair_distance
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from ...metrics import precision_recall_f1_report
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase

logger = logging.getLogger(__name__)


class SpanRelClassificationDecoderMixin(DecoderMixinBase):
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
        
    def exemplify(self, data_entry: dict, training: bool=True, building: bool=True):
        return {'chunk_pairs_obj': ChunkPairs(data_entry, self, training=training, building=building)}
        
    def batchify(self, batch_examples: List[dict]):
        return {'chunk_pairs_objs': [ex['chunk_pairs_obj'] for ex in batch_examples]}
        
    def retrieve(self, batch: Batch):
        return [chunk_pairs_obj.relations for chunk_pairs_obj in batch.chunk_pairs_objs]
        
    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores['micro']['f1']
        
    def inject_chunks_and_build(self, batch: Batch, batch_chunks_pred: List[List[tuple]], device=None):
        """This method is to be invoked outside, as the outside invoker is assumed not to know the structure of `batch`. 
        """
        for chunk_pairs_obj, chunks_pred in zip(batch.chunk_pairs_objs, batch_chunks_pred):
            if not chunk_pairs_obj.is_built:
                chunk_pairs_obj.inject_chunks(chunks_pred)
                chunk_pairs_obj.build(self)
                if device is not None:
                    chunk_pairs_obj.to(device)



class ChunkPairs(TargetWrapper):
    """A wrapper of chunk-pairs with underlying relations. 
    
    Parameters
    ----------
    data_entry: dict
        {'tokens': TokenSequence, 
         'chunks': List[tuple], 
         'relations': List[tuple]}
    
    Notes
    -----
    (1) If `building` is `True`, `data_entry['chunks']` is assumed to be known for both training and evaluation (e.g., pipeline modeling).
        In this case, the negative samples are generated from `data_entry['chunks']`. 
    (2) If `building` is `False`, `data_entry['chunks']` is known for training but not for evaluation (e.g., joint modeling).
        In this case, `inject_chunks` and `build` should be successively invoked, and the negative samples are generated from injected chunks. 
    """
    def __init__(self, data_entry: dict, config: SpanRelClassificationDecoderMixin, training: bool=True, building: bool=True):
        super().__init__(training)
        
        self.chunks = data_entry['chunks'] if training or building else []
        self.relations = data_entry.get('relations', None)
        
        self.is_built = False
        if building:
            self.build(config)
        
        
    def inject_chunks(self, chunks: List[tuple]):
        """Inject chunks from outside, typically from on-the-fly decoding. 
        
        Notes
        -----
        In merged chunks, there may exist two chunks with same spans but different chunk-types. 
        In this case, `ck_label_ids` is crucial as input. 
        """
        assert not self.is_built
        # Do not use ```self.chunks = list(set(self.chunks + chunks))```,
        # which may return non-deterministic order?
        self.chunks = self.chunks + [ck for ck in chunks if ck not in self.chunks]
        
        
    def build(self, config: SpanRelClassificationDecoderMixin):
        """Generate negative samples from `self.chunks` and build up tensors. 
        """
        assert not self.is_built
        self.is_built = True
        
        if self.training:
            pos_chunk_pairs = [(head, tail) for rel_label, head, tail in self.relations]
        else:
            pos_chunk_pairs = []
        
        neg_chunk_pairs = [(head, tail) for head in self.chunks for tail in self.chunks 
                               if (head[1:] != tail[1:] 
                               and (head[0], tail[0]) in config.legal_head_tail_types 
                               and chunk_pair_distance(head, tail) <= config.max_pair_distance
                               and (head, tail) not in pos_chunk_pairs)]
        
        if self.training and len(neg_chunk_pairs) > config.num_neg_relations:
            neg_chunk_pairs = random.sample(neg_chunk_pairs, config.num_neg_relations)
        
        self.chunk_pairs = pos_chunk_pairs + neg_chunk_pairs
        # span_size_ids / ck_label_ids: (num_pairs, 2)
        # Note: size_id = size - 1
        self.span_size_ids = torch.tensor([[h_end-h_start-1, t_end-t_start-1] 
                                               for (h_label, h_start, h_end), (t_label, t_start, t_end) 
                                               in self.chunk_pairs], 
                                          dtype=torch.long)
        self.span_size_ids.masked_fill_(self.span_size_ids>=config.max_span_size, config.max_span_size-1)
        
        self.ck_label_ids = torch.tensor([[config.ck_label2idx[h_label], config.ck_label2idx[t_label]] 
                                              for (h_label, h_start, h_end), (t_label, t_start, t_end) 
                                              in self.chunk_pairs], 
                                         dtype=torch.long)
        
        if self.relations is not None:
            chunk_pair2rel_label = {(head, tail): rel_label for rel_label, head, tail in self.relations}
            rel_labels = [chunk_pair2rel_label.get((head, tail), config.rel_none_label) for head, tail in self.chunk_pairs]
            self.rel_label_ids = torch.tensor([config.rel_label2idx[rel_label] for rel_label in rel_labels], dtype=torch.long)




class SpanRelClassificationDecoderConfig(SingleDecoderConfigBase, SpanRelClassificationDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        self.num_neg_relations = kwargs.pop('num_neg_relations', 100)
        self.max_span_size = kwargs.pop('max_span_size', 10)
        self.max_pair_distance = kwargs.pop('max_pair_distance', 100)
        self.ck_size_emb_dim = kwargs.pop('ck_size_emb_dim', 25)
        self.ck_label_emb_dim = kwargs.pop('ck_label_emb_dim', 25)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        
        self.ck_none_label = kwargs.pop('ck_none_label', '<none>')
        self.idx2ck_label = kwargs.pop('idx2ck_label', None)
        self.rel_none_label = kwargs.pop('rel_none_label', '<none>')
        self.idx2rel_label = kwargs.pop('idx2rel_label', None)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])
    
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'agg_mode', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    def build_vocab(self, *partitions):
        ck_counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2ck_label = [self.ck_none_label] + list(ck_counter.keys())
        rel_counter = Counter(label for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.idx2rel_label = [self.rel_none_label] + list(rel_counter.keys())
        ht_counter = Counter((head[0], tail[0]) for data in partitions for entry in data for label, head, tail in entry['relations'])
        self.legal_head_tail_types = set(list(ht_counter.keys()))
        
        dist_counter = Counter(chunk_pair_distance(head, tail) for data in partitions for entry in data for label, head, tail in entry['relations'])
        num_pairs = sum(dist_counter.values())
        num_oov_pairs = sum(num for dist, num in dist_counter.items() if dist > self.max_pair_distance)
        if num_oov_pairs > 0:
            logger.warning(f"OOV positive pairs: {num_oov_pairs} ({num_oov_pairs/num_pairs*100:.2f}%)")
        
        
    def instantiate(self):
        return SpanRelClassificationDecoder(self)




class SpanRelClassificationDecoder(DecoderBase, SpanRelClassificationDecoderMixin):
    def __init__(self, config: SpanRelClassificationDecoderConfig):
        super().__init__()
        self.num_neg_relations = config.num_neg_relations
        self.max_span_size = config.max_span_size
        self.max_pair_distance = config.max_pair_distance
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.rel_none_label = config.rel_none_label
        self.idx2rel_label = config.idx2rel_label
        self.legal_head_tail_types = config.legal_head_tail_types
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        # Trainable context vector for overlapping chunks
        self.zero_context = torch.nn.Parameter(torch.zeros(config.in_dim))
        
        if config.ck_size_emb_dim > 0:
            self.ck_size_embedding = torch.nn.Embedding(config.max_span_size, config.ck_size_emb_dim)
            reinit_embedding_(self.ck_size_embedding)
        if config.ck_label_emb_dim > 0:
            self.ck_label_embedding = torch.nn.Embedding(config.ck_voc_dim, config.ck_label_emb_dim)
            reinit_embedding_(self.ck_label_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim*3+config.ck_size_emb_dim*2+config.ck_label_emb_dim*2, config.rel_voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for k in range(full_hidden.size(0)):
            if len(batch.chunk_pairs_objs[k].chunk_pairs) == 0:
                logits = torch.empty(0, self.hid2logit.out_features, device=full_hidden.device)
                
            else:
                head_hidden, tail_hidden, contexts = [], [], []
                for (h_label, h_start, h_end), (t_label, t_start, t_end) in batch.chunk_pairs_objs[k].chunk_pairs:
                    head_hidden.append(full_hidden[k, h_start:h_end])
                    tail_hidden.append(full_hidden[k, t_start:t_end])
                    
                    if h_end < t_start:
                        contexts.append(full_hidden[k, h_end:t_start])
                    elif t_end < h_start:
                        contexts.append(full_hidden[k, t_end:h_start])
                    else:
                        contexts.append(self.zero_context.unsqueeze(0))
                
                # head_hidden: (num_pairs, span_size, hid_dim) -> (num_pairs, hid_dim)
                head_mask = seq_lens2mask(torch.tensor([h.size(0) for h in head_hidden], dtype=torch.long, device=full_hidden.device))
                head_hidden = torch.nn.utils.rnn.pad_sequence(head_hidden, batch_first=True, padding_value=0.0)
                head_hidden = self.aggregating(self.dropout(head_hidden), mask=head_mask)
                
                # tail_hidden: (num_pairs, span_size, hid_dim) -> (num_pairs, hid_dim)
                tail_mask = seq_lens2mask(torch.tensor([h.size(0) for h in tail_hidden], dtype=torch.long, device=full_hidden.device))
                tail_hidden = torch.nn.utils.rnn.pad_sequence(tail_hidden, batch_first=True, padding_value=0.0)
                tail_hidden = self.aggregating(self.dropout(tail_hidden), mask=tail_mask)
                
                # contexts: (num_pairs, span_size, hid_dim) -> (num_pairs, hid_dim)
                ctx_mask = seq_lens2mask(torch.tensor([c.size(0) for c in contexts], dtype=torch.long, device=full_hidden.device))
                contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True, padding_value=0.0)
                contexts = self.aggregating(self.dropout(contexts), mask=ctx_mask)
                
                # pair_hidden: (num_pairs, hid_dim*3)
                pair_hidden = torch.cat([head_hidden, tail_hidden, contexts], dim=-1)
                
                if hasattr(self, 'ck_size_embedding'):
                    # ck_size_embedded: (num_pairs, 2, emb_dim) -> (num_pairs, emb_dim*2)
                    ck_size_embedded = self.ck_size_embedding(batch.chunk_pairs_objs[k].span_size_ids).flatten(start_dim=1)
                    pair_hidden = torch.cat([pair_hidden, self.dropout(ck_size_embedded)], dim=-1)
                
                if hasattr(self, 'ck_label_embedding'):
                    # ck_label_embedded: (num_pairs, 2, emb_dim) -> (num_pairs, emb_dim*2)
                    ck_label_embedded = self.ck_label_embedding(batch.chunk_pairs_objs[k].ck_label_ids).flatten(start_dim=1)
                    pair_hidden = torch.cat([pair_hidden, self.dropout(ck_label_embedded)], dim=-1)
                
                logits = self.hid2logit(pair_hidden)
            
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        losses = [self.criterion(batch_logits[k], batch.chunk_pairs_objs[k].rel_label_ids) 
                      if len(batch.chunk_pairs_objs[k].chunk_pairs) > 0
                      else torch.tensor(0.0, device=full_hidden.device)
                      for k in range(full_hidden.size(0))]
        return torch.stack(losses)
    
    
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_relations = []
        for k in range(full_hidden.size(0)):
            if len(batch.chunk_pairs_objs[k].chunk_pairs) > 0:
                rel_labels = [self.idx2rel_label[i] for i in batch_logits[k].argmax(dim=-1).cpu().tolist()]
                relations = [(rel_label, head, tail) 
                                 for rel_label, (head, tail)
                                 in zip(rel_labels, batch.chunk_pairs_objs[k].chunk_pairs) 
                                 if rel_label != self.rel_none_label]
            else:
                relations = []
            batch_relations.append(relations)
        
        return batch_relations
