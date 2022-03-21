# -*- coding: utf-8 -*-
from typing import Dict
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...utils.chunk import detect_overlapping_level, filter_clashed_by_priority
from ...nn.modules import CombinedDropout, SoftLabelCrossEntropyLoss
from ...nn.init import reinit_embedding_, reinit_layer_
from .base import DecoderMixinBase, SingleDecoderConfigBase, DecoderBase
from .boundaries import Boundaries, _spans_from_diagonals
from .boundary_selection import BoundarySelectionDecoderMixin as SpecificSpanClsDecoderMixin

logger = logging.getLogger(__name__)



class SpecificSpanClsDecoderConfig(SingleDecoderConfigBase, SpecificSpanClsDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.2, 0.0, 0.0))
        
        # self.max_len = kwargs.pop('max_len', None)
        
        # Note: The spans with sizes longer than `max_span_size` will be masked/ignored in both training and inference. 
        # Hence, these spans will never be recalled in testing. 
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 20)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 0.995)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        self.hard_neg_sampling_rate = kwargs.pop('hard_neg_sampling_rate', 1.0)
        self.hard_neg_sampling_rate = max(self.hard_neg_sampling_rate, self.neg_sampling_rate)
        self.hard_neg_sampling_size = kwargs.pop('hard_neg_sampling_size', 5)
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.overlapping_level = kwargs.pop('overlapping_level', None)
        
        # Boundary smoothing epsilon
        self.sb_epsilon = kwargs.pop('sb_epsilon', 0.0)
        self.sb_size = kwargs.pop('sb_size', 1)
        self.sb_adj_factor = kwargs.pop('sb_adj_factor', 1.0)
        super().__init__(**kwargs)
        
    @property
    def name(self):
        return self.criterion
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'criterion']}
        return self._repr_non_config_attrs(repr_attr_dict)
        
    @property
    def criterion(self):
        if self.sb_epsilon > 0:
            return f"SB({self.sb_epsilon:.2f}, {self.sb_size})"
        else:
            return super().criterion
        
    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith(('sb', 'sl')):
            # For boundary/label smoothing, the `Boundaries` object has been accordingly changed; 
            # hence, do not use `SmoothLabelCrossEntropyLoss`
            return SoftLabelCrossEntropyLoss(**kwargs)
        else:
            return super().instantiate_criterion(**kwargs)
        
        
    def build_vocab(self, *partitions):
        counter = Counter(label for data in partitions for entry in data for label, start, end in entry['chunks'])
        self.idx2label = [self.none_label] + list(counter.keys())
        
        self.overlapping_level = max(detect_overlapping_level(entry['chunks']) for data in partitions for entry in data)
        logger.info(f"Overlapping level: {self.overlapping_level}")
        
        # Allow directly setting `max_span_size`
        if self.max_span_size is None:
            # Calculate `max_span_size` according to data
            span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
            if self.max_span_size_cov_rate >= 1:
                span_size_cov = max(span_sizes)
            else:
                span_size_cov = math.ceil(numpy.quantile(span_sizes, self.max_span_size_cov_rate))
            self.max_span_size = min(span_size_cov, self.max_span_size_ceiling)
        logger.warning(f"The `max_span_size` is set to {self.max_span_size}")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
        
        
    def instantiate(self):
        return SpecificSpanClsDecoder(self)



class SpecificSpanClsDecoder(DecoderBase, SpecificSpanClsDecoderMixin):
    def __init__(self, config: SpecificSpanClsDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.overlapping_level = config.overlapping_level
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_span_size, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        # full_hidden: (batch, step, hid_dim)
        # query_hidden: (batch, step-k+1, hid_dim)
        all_hidden = [full_hidden] + list(all_query_hidden.values())
        
        batch_logits = []
        for i, curr_len in enumerate(batch.seq_lens.cpu().tolist()):
            curr_max_span_size = min(self.max_span_size, curr_len)
            
            # (curr_len-k+1, hid_dim) -> (\sum_k curr_len-k+1, hid_dim)
            span_hidden = torch.cat([all_hidden[k-1][i, :curr_len-k+1] for k in range(1, curr_max_span_size+1)], dim=0)
            
            if hasattr(self, 'size_embedding'):
                span_size_ids = [k-1 for k in range(1, curr_max_span_size+1) for _ in range(curr_len-k+1)]
                span_size_ids = torch.tensor(span_size_ids, dtype=torch.long, device=full_hidden.device)
                # size_embedded: (\sum_k curr_len-k+1, emb_dim)
                size_embedded = self.size_embedding(span_size_ids)
                span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)
            
            # (\sum_k curr_len-k+1, logit_dim)
            logits = self.hid2logit(self.dropout(span_hidden))
            batch_logits.append(logits)
        
        return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
        batch_label_ids = []
        for boundaries_obj, curr_len in zip(batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            curr_max_span_size = min(self.max_span_size, curr_len)
            
            # label_ids: (curr_len-k+1, ) -> (\sum_k curr_len-k+1, )
            label_ids = torch.cat([boundaries_obj.boundary2label_id.diagonal(offset=k-1) for k in range(1, curr_max_span_size+1)], dim=-1)
            if label_ids.dim() == 2:
                # label_ids: (\sum_k curr_len-k+1, logit_dim)
                label_ids = label_ids.permute(1, 0)
            batch_label_ids.append(label_ids)
        
        losses = [self.criterion(logits, label_ids) for logits, label_ids in zip(batch_logits, batch_label_ids)]
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor, all_query_hidden: Dict[int, torch.Tensor]):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)
        
        batch_chunks = []
        for logits, curr_len in zip(batch_logits, batch.seq_lens.cpu().tolist()):
            curr_max_span_size = min(self.max_span_size, curr_len)
            
            confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
            chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_diagonals(curr_len, curr_max_span_size)) if label != self.none_label]
            confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            assert len(confidences) == len(chunks)
            
            # Sort chunks from high to low confidences
            chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)
            
            batch_chunks.append(chunks)
        return batch_chunks
