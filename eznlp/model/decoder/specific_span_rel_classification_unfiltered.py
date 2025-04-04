# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter
from typing import Dict, List

import numpy
import torch

from ...metrics import precision_recall_f1_report
from ...nn.init import reinit_layer_
from ...nn.modules import BiAffineFusor, CombinedDropout
from ...wrapper import Batch
from ..encoder import EncoderConfig
from .base import DecoderBase, DecoderMixinBase, SingleDecoderConfigBase
from .boundaries import (
    MAX_SIZE_ID_COV_RATE,
    DiagBoundariesPairs,
    _span_pairs_from_diagonals,
)

logger = logging.getLogger(__name__)


class DiagBoundariesPairsDecoderMixin(DecoderMixinBase):
    """A `Mixin` for relation extraction."""

    @property
    def idx2label(self):
        return self._idx2label

    @idx2label.setter
    def idx2label(self, idx2label: List[str]):
        self._idx2label = idx2label
        self.label2idx = (
            {l: i for i, l in enumerate(idx2label)} if idx2label is not None else None
        )

    @property
    def voc_dim(self):
        return len(self.label2idx)

    @property
    def none_idx(self):
        return self.label2idx[self.none_label]

    @property
    def idx2ck_label(self):
        return self._idx2ck_label

    @idx2ck_label.setter
    def idx2ck_label(self, idx2ck_label: List[str]):
        self._idx2ck_label = idx2ck_label
        self.ck_label2idx = (
            {l: i for i, l in enumerate(idx2ck_label)}
            if idx2ck_label is not None
            else None
        )

    @property
    def ck_voc_dim(self):
        return len(self.ck_label2idx)

    @property
    def ck_none_idx(self):
        return self.ck_label2idx[self.ck_none_label]

    def exemplify(self, entry: dict, training: bool = True):
        return {"dbp_obj": DiagBoundariesPairs(entry, self, training=training)}

    def batchify(self, batch_examples: List[dict]):
        return {"dbp_objs": [ex["dbp_obj"] for ex in batch_examples]}

    def retrieve(self, batch: Batch):
        return [dbp_obj.relations for dbp_obj in batch.dbp_objs]

    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores["micro"]["f1"]

    def _filter(self, relations: List[tuple]):
        relations = [
            (label, head, tail)
            for label, head, tail in relations
            if (
                (label, head[0], tail[0]) in self.existing_rht_labels
                and (self.existing_self_rel or head[1:] != tail[1:])
            )
        ]
        return relations


class UnfilteredSpecificSpanRelClsDecoderConfig(
    SingleDecoderConfigBase, DiagBoundariesPairsDecoderMixin
):
    def __init__(self, **kwargs):
        self.fusing_mode = kwargs.pop("fusing_mode", "concat")
        self.reduction = kwargs.pop(
            "reduction",
            EncoderConfig(
                arch="FFN",
                hid_dim=150,
                num_layers=1,
                in_drop_rates=(0.0, 0.0, 0.0),
                hid_drop_rate=0.0,
            ),
        )

        self.in_drop_rates = kwargs.pop("in_drop_rates", (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop("hid_drop_rates", (0.2, 0.0, 0.0))

        self.min_span_size = kwargs.pop("min_span_size", 2)
        assert self.min_span_size in (1, 2)
        self.max_span_size_ceiling = kwargs.pop("max_span_size_ceiling", 20)
        self.max_span_size_cov_rate = kwargs.pop("max_span_size_cov_rate", 0.995)
        self.max_span_size = kwargs.pop("max_span_size", None)

        self.neg_sampling_rate = kwargs.pop("neg_sampling_rate", 1.0)

        self.none_label = kwargs.pop("none_label", "<none>")
        self.idx2label = kwargs.pop("idx2label", None)
        self.ck_none_label = kwargs.pop("ck_none_label", "<none>")
        self.idx2ck_label = kwargs.pop("idx2ck_label", None)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name_sep.join([self.fusing_mode, self.criterion])

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key)
            for key in [
                "in_dim",
                "in_drop_rates",
                "hid_drop_rates",
                "fusing_mode",
                "criterion",
            ]
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    @property
    def in_dim(self):
        return self.reduction.in_dim

    @in_dim.setter
    def in_dim(self, dim: int):
        if dim is not None:
            self.reduction.in_dim = dim

    def build_vocab(self, *partitions):
        counter = Counter(
            label
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        )
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())

        # Calculate `max_span_size` according to data
        span_sizes = [
            end - start
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        ]
        # Allow directly setting `max_span_size`
        if self.max_span_size is None:
            if self.max_span_size_cov_rate >= 1:
                span_size_cov = max(span_sizes)
            else:
                span_size_cov = math.ceil(
                    numpy.quantile(span_sizes, self.max_span_size_cov_rate)
                )
            self.max_span_size = min(span_size_cov, self.max_span_size_ceiling)
        self.max_size_id = (
            min(
                math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)),
                self.max_span_size,
            )
            - 1
        )
        logger.warning(f"The `max_span_size` is set to {self.max_span_size}")

        size_counter = Counter(
            end - start
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        )
        num_spans = sum(size_counter.values())
        num_oov_spans = sum(
            num for size, num in size_counter.items() if size > self.max_span_size
        )
        if num_oov_spans > 0:
            logger.warning(
                f"OOV positive spans: {num_oov_spans} ({num_oov_spans/num_spans*100:.2f}%)"
            )

        counter = Counter(
            label
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        )
        self.idx2label = [self.none_label] + list(counter.keys())

        counter = Counter(
            (label, head[0], tail[0])
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        )
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_self_rel = any(
            head[1:] == tail[1:]
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        )

    def instantiate(self):
        return UnfilteredSpecificSpanRelClsDecoder(self)


class UnfilteredSpecificSpanRelClsDecoder(DecoderBase, DiagBoundariesPairsDecoderMixin):
    def __init__(self, config: UnfilteredSpecificSpanRelClsDecoderConfig):
        super().__init__()
        self.fusing_mode = config.fusing_mode
        self.min_span_size = config.min_span_size
        self.max_span_size = config.max_span_size
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.existing_rht_labels = config.existing_rht_labels
        self.existing_self_rel = config.existing_self_rel

        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)

        if config.fusing_mode.lower().startswith("concat"):
            self.hid2logit = torch.nn.Linear(config.in_dim * 2, config.voc_dim)
            reinit_layer_(self.hid2logit, "sigmoid")

        elif config.fusing_mode.lower().startswith("affine"):
            self.reduction_head = config.reduction.instantiate()
            self.reduction_tail = config.reduction.instantiate()
            self.hid2logit = BiAffineFusor(config.reduction.out_dim, config.voc_dim)

        self.criterion = config.instantiate_criterion(reduction="sum")

    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling."""
        for dbp_obj, chunks_pred in zip(batch.dbp_objs, batch_chunks_pred):
            if dbp_obj.chunks_pred is None:
                dbp_obj.chunks_pred = chunks_pred
                dbp_obj.build(self)
                if self.fusing_mode.startswith("concat"):
                    dbp_obj.to(self.hid2logit.weight.device)
                elif self.fusing_mode.lower().startswith("affine"):
                    dbp_obj.to(self.hid2logit.U.device)

    def get_logits(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        all_query_hidden: Dict[int, torch.Tensor],
    ):
        # full_hidden: (batch, step, hid_dim)
        # query_hidden: (batch, step-k+1, hid_dim)
        if self.min_span_size == 1:
            all_hidden = list(all_query_hidden.values())
        else:
            all_hidden = [full_hidden] + list(all_query_hidden.values())

        batch_logits = []
        for i, curr_len in enumerate(batch.seq_lens.cpu().tolist()):
            # (curr_len-k+1, hid_dim) -> (num_spans = \sum_k curr_len-k+1, hid_dim)
            span_hidden = torch.cat(
                [
                    all_hidden[k - 1][i, : curr_len - k + 1]
                    for k in range(1, min(self.max_span_size, curr_len) + 1)
                ],
                dim=0,
            )

            num_spans = span_hidden.size(0)
            head_hidden = (
                self.in_dropout(span_hidden).unsqueeze(1).expand(-1, num_spans, -1)
            )
            tail_hidden = (
                self.in_dropout(span_hidden).unsqueeze(0).expand(num_spans, -1, -1)
            )

            if self.fusing_mode.startswith("concat"):
                # hidden_cat: (num_spans, num_spans, hid_dim*2)
                hidden_cat = torch.cat([head_hidden, tail_hidden], dim=-1)
                # logits: (num_chunks, num_chunks, logit_dim)
                logits = self.hid2logit(hidden_cat)

            elif self.fusing_mode.lower().startswith("affine"):
                reduced_head = self.reduction_head(head_hidden)
                reduced_tail = self.reduction_tail(tail_hidden)
                logits = self.hid2logit(
                    self.hid_dropout(reduced_head), self.hid_dropout(reduced_tail)
                )

            batch_logits.append(logits)

        return batch_logits

    def forward(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        all_query_hidden: Dict[int, torch.Tensor],
    ):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)

        losses = []
        for logits, dbp_obj in zip(batch_logits, batch.dbp_objs):
            # label_ids: (num_spans, num_spans) or (num_spans, num_spans, voc_dim)
            label_ids = dbp_obj.dbp2label_id
            if hasattr(dbp_obj, "non_mask"):
                non_mask = dbp_obj.non_mask
                logits, label_ids = logits[non_mask], label_ids[non_mask]
            else:
                logits, label_ids = logits.flatten(end_dim=1), label_ids.flatten(
                    end_dim=1
                )

            loss = self.criterion(logits, label_ids)
            losses.append(loss)
        return torch.stack(losses)

    def decode(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        all_query_hidden: Dict[int, torch.Tensor],
    ):
        batch_logits = self.get_logits(batch, full_hidden, all_query_hidden)

        batch_relations = []
        for logits, dbp_obj, curr_len in zip(
            batch_logits, batch.dbp_objs, batch.seq_lens.cpu().tolist()
        ):
            confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
            labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]

            relations = []
            for label, ((h_start, h_end), (t_start, t_end)) in zip(
                labels, _span_pairs_from_diagonals(curr_len, self.max_span_size)
            ):
                head = (
                    dbp_obj.span2ck_label.get((h_start, h_end), self.ck_none_label),
                    h_start,
                    h_end,
                )
                tail = (
                    dbp_obj.span2ck_label.get((t_start, t_end), self.ck_none_label),
                    t_start,
                    t_end,
                )
                if (
                    label != self.none_label
                    and head[0] != self.ck_none_label
                    and tail[0] != self.ck_none_label
                ):
                    relations.append((label, head, tail))
            relations = self._filter(relations)
            batch_relations.append(relations)

        return batch_relations
