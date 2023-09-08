# -*- coding: utf-8 -*-
from collections import Counter
import logging
import math
import numpy
import torch

from ...wrapper import Batch
from ...utils.chunk import detect_overlapping_level, filter_clashed_by_priority
from ...nn.modules import SequencePooling, SequenceAttention, CombinedDropout, SoftLabelCrossEntropyLoss, MultiKernelMaxMeanDiscrepancyLoss
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from .base import SingleDecoderConfigBase, DecoderBase
from .boundaries import Boundaries, MAX_SIZE_ID_COV_RATE, _spans_from_diagonals
from .boundary_selection import BoundariesDecoderMixin

logger = logging.getLogger(__name__)



class SpanClassificationDecoderConfig(SingleDecoderConfigBase, BoundariesDecoderMixin):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop('in_drop_rates', (0.5, 0.0, 0.0))
        
        # Difference between `max_span_size` and `max_size_id`:
        # (1) The spans with sizes longer than `max_span_size` will be masked/ignored in both training and inference. 
        # Hence, these spans will never be recalled in testing. 
        # (2) `max_size_id` is only used for creating `size_embedding`. The spans with sizes longer than `max_size_id+1` 
        # share a same size embedding vector; such spans are used for training and inference. 
        self.max_span_size_ceiling = kwargs.pop('max_span_size_ceiling', 20)
        self.max_span_size_cov_rate = kwargs.pop('max_span_size_cov_rate', 0.995)
        self.max_span_size = kwargs.pop('max_span_size', None)
        self.max_len = kwargs.pop('max_len', None)
        self.size_emb_dim = kwargs.pop('size_emb_dim', 25)
        
        self.neg_sampling_rate = kwargs.pop('neg_sampling_rate', 1.0)
        self.neg_sampling_power_decay = kwargs.pop('neg_sampling_power_decay', 0.0)  # decay = 0.5, 1.0
        self.neg_sampling_surr_rate = kwargs.pop('neg_sampling_surr_rate', 0.0)
        self.neg_sampling_surr_size = kwargs.pop('neg_sampling_surr_size', 5)
        
        # Spans internal (i.e., nested) / external spans to gold entities
        self.nested_sampling_rate = kwargs.pop('nested_sampling_rate', 1.0)
        self.inex_mkmmd_lambda = kwargs.pop('inex_mkmmd_lambda', 0.0)
        self.inex_mkmmd_num_kernels = kwargs.pop('inex_mkmmd_num_kernels', 5)
        self.inex_mkmmd_multiplier = kwargs.pop('inex_mkmmd_multiplier', 2.0)
        self.inex_mkmmd_xsample = kwargs.pop('inex_mkmmd_xsample', False)
        
        self.agg_mode = kwargs.pop('agg_mode', 'max_pooling')
        
        self.none_label = kwargs.pop('none_label', '<none>')
        self.idx2label = kwargs.pop('idx2label', None)
        self.overlapping_level = kwargs.pop('overlapping_level', None)
        self.chunk_priority = kwargs.pop('chunk_priority', 'confidence')
        
        # Boundary smoothing epsilon
        self.sb_epsilon = kwargs.pop('sb_epsilon', 0.0)
        self.sb_size = kwargs.pop('sb_size', 1)
        self.sb_adj_factor = kwargs.pop('sb_adj_factor', 1.0)
        super().__init__(**kwargs)
        
        
    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])
        
    def __repr__(self):
        repr_attr_dict = {key: getattr(self, key) for key in ['in_dim', 'in_drop_rates', 'agg_mode', 'criterion']}
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
        
        # Calculate `max_span_size` according to data
        span_sizes = [end-start for data in partitions for entry in data for label, start, end in entry['chunks']]
        # Allow directly setting `max_span_size`
        if self.max_span_size is None:
            if self.max_span_size_cov_rate >= 1:
                span_size_cov = max(span_sizes)
            else:
                span_size_cov = math.ceil(numpy.quantile(span_sizes, self.max_span_size_cov_rate))
            self.max_span_size = min(span_size_cov, self.max_span_size_ceiling)
        self.max_size_id = min(math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)), self.max_span_size) - 1
        logger.warning(f"The `max_span_size` is set to {self.max_span_size}")
        
        size_counter = Counter(end-start for data in partitions for entry in data for label, start, end in entry['chunks'])
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(num for size, num in size_counter.items() if size > self.max_span_size)
        if num_oov_spans > 0:
            logger.warning(f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)")
        
        self.max_len = max(len(data_entry['tokens']) for data in partitions for data_entry in data)
        
        
    def instantiate(self):
        return SpanClassificationDecoder(self)



class SpanClassificationDecoder(DecoderBase, BoundariesDecoderMixin):
    def __init__(self, config: SpanClassificationDecoderConfig):
        super().__init__()
        self.max_span_size = config.max_span_size
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.overlapping_level = config.overlapping_level
        self.chunk_priority = config.chunk_priority
        self.inex_mkmmd_lambda = config.inex_mkmmd_lambda
        
        if config.agg_mode.lower().endswith('_pooling'):
            self.aggregating = SequencePooling(mode=config.agg_mode.replace('_pooling', ''))
        elif config.agg_mode.lower().endswith('_attention'):
            self.aggregating = SequenceAttention(config.in_dim, scoring=config.agg_mode.replace('_attention', ''))
        
        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(config.max_size_id+1, config.size_emb_dim)
            reinit_embedding_(self.size_embedding)
        
        self.register_buffer('_span_size_ids', torch.arange(config.max_len) - torch.arange(config.max_len).unsqueeze(-1))
        self._span_size_ids.masked_fill_(self._span_size_ids < 0, 0)
        self._span_size_ids.masked_fill_(self._span_size_ids > config.max_size_id, config.max_size_id)
        
        if config.inex_mkmmd_lambda > 0:
            self.inex_mkmmd = MultiKernelMaxMeanDiscrepancyLoss(config.inex_mkmmd_num_kernels, config.inex_mkmmd_multiplier)
        
        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim+config.size_emb_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, 'sigmoid')
        
        self.criterion = config.instantiate_criterion(reduction='sum')
        
        
    def _get_diagonal_span_size_ids(self, seq_len: int):
        span_size_ids = self._span_size_ids[:seq_len, :seq_len]
        return torch.cat([span_size_ids.diagonal(offset=k-1) for k in range(1, min(self.max_span_size, seq_len)+1)], dim=-1)
        
        
    def get_logits(self, batch: Batch, full_hidden: torch.Tensor, return_states: bool=False):
        # full_hidden: (batch, step, hid_dim)
        batch_logits, batch_states = [], []
        for i, curr_len in enumerate(batch.seq_lens.cpu().tolist()):
            # span_hidden: (num_spans, span_size, hid_dim) -> (num_spans, hid_dim)
            span_hidden = [full_hidden[i, start:end] for start, end in _spans_from_diagonals(curr_len, self.max_span_size)]
            span_mask = seq_lens2mask(torch.tensor([h.size(0) for h in span_hidden], dtype=torch.long, device=full_hidden.device))
            span_hidden = torch.nn.utils.rnn.pad_sequence(span_hidden, batch_first=True, padding_value=0.0)
            span_hidden = self.aggregating(self.dropout(span_hidden), mask=span_mask)
            
            if hasattr(self, 'size_embedding'):
                # size_embedded: (num_spans, emb_dim)
                size_embedded = self.size_embedding(self._get_diagonal_span_size_ids(curr_len))
                span_hidden = torch.cat([span_hidden, self.dropout(size_embedded)], dim=-1)
            
            logits = self.hid2logit(span_hidden)
            batch_logits.append(logits)
            batch_states.append({'span_hidden': span_hidden})
        
        if return_states:
            return batch_logits, batch_states
        else:
            return batch_logits
        
        
    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits, batch_states = self.get_logits(batch, full_hidden, return_states=True)
        
        losses = []
        for logits, boundaries_obj, curr_len in zip(batch_logits, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            # label_ids: (num_spans = \sum_k curr_len-k+1, ) or (num_spans = \sum_k curr_len-k+1, logit_dim)
            label_ids = boundaries_obj.diagonal_label_ids
            if hasattr(boundaries_obj, 'non_mask'):
                non_mask = boundaries_obj.diagonal_non_mask
                logits, label_ids = logits[non_mask], label_ids[non_mask]
            
            loss = self.criterion(logits, label_ids)
            losses.append(loss)
        
        if hasattr(self, 'inex_mkmmd'):
            aux_losses = []
            for states, boundaries_obj, curr_len in zip(batch_states, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
                nest_non_mask = boundaries_obj.diagonal_nest_non_mask
                aux_loss = self.inex_mkmmd(states['span_hidden'][nest_non_mask], states['span_hidden'][~nest_non_mask])
                aux_losses.append(aux_loss)
            losses = [l+al for l, al in zip(losses, aux_losses)]
        
        return torch.stack(losses)
        
        
    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)
        
        batch_chunks = []
        for logits, boundaries_obj, curr_len in zip(batch_logits, batch.boundaries_objs, batch.seq_lens.cpu().tolist()):
            confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.cpu().tolist()]
            chunks = [(label, start, end) for label, (start, end) in zip(labels, _spans_from_diagonals(curr_len, self.max_span_size)) if label != self.none_label]
            confidences = [conf for label, conf in zip(labels, confidences.cpu().tolist()) if label != self.none_label]
            assert len(confidences) == len(chunks)
            
            if hasattr(boundaries_obj, 'sub2ori_idx'):
                is_valid = [isinstance(boundaries_obj.sub2ori_idx[start], int) and isinstance(boundaries_obj.sub2ori_idx[end], int) for label, start, end in chunks]
                confidences = [conf for conf, is_v in zip(confidences, is_valid) if is_v]
                chunks = [ck for ck, is_v in zip(chunks, is_valid) if is_v]
            
            if self.chunk_priority.lower().startswith('len'):
                # Sort chunks by lengths: long -> short 
                chunks = sorted(chunks, key=lambda ck: ck[2]-ck[1], reverse=True)
            else:
                # Sort chunks by confidences: high -> low 
                chunks = [ck for _, ck in sorted(zip(confidences, chunks), reverse=True)]
            chunks = filter_clashed_by_priority(chunks, allow_level=self.overlapping_level)
            
            batch_chunks.append(chunks)
        return batch_chunks
