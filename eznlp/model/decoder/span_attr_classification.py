# -*- coding: utf-8 -*-
import logging
import math
from collections import Counter
from typing import List

import numpy
import torch

from ...metrics import precision_recall_f1_report
from ...nn.functional import seq_lens2mask
from ...nn.init import reinit_embedding_, reinit_layer_
from ...nn.modules import CombinedDropout, SequenceAttention, SequencePooling
from ...wrapper import Batch
from ..encoder import EncoderConfig
from .base import DecoderBase, DecoderMixinBase, SingleDecoderConfigBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import ChunkSingles

logger = logging.getLogger(__name__)


class ChunkSinglesDecoderMixin(DecoderMixinBase):
    """A `Mixin` for attribute extraction."""

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
        return {"cs_obj": ChunkSingles(entry, self, training=training)}

    def batchify(self, batch_examples: List[dict]):
        return {"cs_objs": [ex["cs_obj"] for ex in batch_examples]}

    def retrieve(self, batch: Batch):
        return [cs_obj.attributes for cs_obj in batch.cs_objs]

    def evaluate(self, y_gold: List[List[tuple]], y_pred: List[List[tuple]]):
        scores, ave_scores = precision_recall_f1_report(y_gold, y_pred)
        return ave_scores["micro"]["f1"]

    def _filter(self, attributes: List[tuple]):
        attributes = [
            (label, chunk)
            for label, chunk in attributes
            if (
                not self.check_ac_labels
                or ((label, chunk[0]) in self.existing_ac_labels)
            )
        ]
        return attributes


class SpanAttrClassificationDecoderConfig(
    SingleDecoderConfigBase, ChunkSinglesDecoderMixin
):
    def __init__(self, **kwargs):
        self.size_emb_dim = kwargs.pop("size_emb_dim", 0)
        self.label_emb_dim = kwargs.pop("label_emb_dim", 0)
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

        self.neg_sampling_rate = kwargs.pop("neg_sampling_rate", 1.0)
        # self.neg_sampling_power_decay = kwargs.pop('neg_sampling_power_decay', 0.0)  # decay = 0.5, 1.0
        # self.neg_sampling_surr_rate = kwargs.pop('neg_sampling_surr_rate', 0.0)
        # self.neg_sampling_surr_size = kwargs.pop('neg_sampling_surr_size', 5)

        self.agg_mode = kwargs.pop("agg_mode", "max_pooling")
        self.ck_loss_weight = kwargs.pop("ck_loss_weight", 0)

        self.check_ac_labels = kwargs.pop("check_ac_labels", False)
        self.none_label = kwargs.pop("none_label", "<none>")
        self.idx2label = kwargs.pop("idx2label", None)
        self.ck_none_label = kwargs.pop("ck_none_label", "<none>")
        self.idx2ck_label = kwargs.pop("idx2ck_label", None)

        # Change the default as multi-label classification
        kwargs["multilabel"] = kwargs.pop("multilabel", True)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key)
            for key in [
                "in_dim",
                "in_drop_rates",
                "hid_drop_rates",
                "agg_mode",
                "criterion",
            ]
        }
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
        counter = Counter(
            label
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        )
        self.idx2ck_label = [self.ck_none_label] + list(counter.keys())

        span_sizes = [
            end - start
            for data in partitions
            for entry in data
            for label, start, end in entry["chunks"]
        ]
        self.max_size_id = (
            math.ceil(numpy.quantile(span_sizes, MAX_SIZE_ID_COV_RATE)) - 1
        )

        counter = Counter(
            label
            for data in partitions
            for entry in data
            for label, chunk in entry["attributes"]
        )
        self.idx2label = [self.none_label] + list(counter.keys())

        counter = Counter(
            (label, chunk[0])
            for data in partitions
            for entry in data
            for label, chunk in entry["attributes"]
        )
        self.existing_ac_labels = set(list(counter.keys()))

    def instantiate(self):
        return SpanAttrClassificationDecoder(self)


class SpanAttrClassificationDecoder(DecoderBase, ChunkSinglesDecoderMixin):
    def __init__(self, config: SpanAttrClassificationDecoderConfig):
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

        if config.agg_mode.lower().endswith("_pooling"):
            self.aggregating = SequencePooling(
                mode=config.agg_mode.replace("_pooling", "")
            )
        elif config.agg_mode.lower().endswith("_attention"):
            self.aggregating = SequenceAttention(
                config.in_dim, scoring=config.agg_mode.replace("_attention", "")
            )

        if config.size_emb_dim > 0:
            self.size_embedding = torch.nn.Embedding(
                config.max_size_id + 1, config.size_emb_dim
            )
            reinit_embedding_(self.size_embedding)

        if config.label_emb_dim > 0:
            self.label_embedding = torch.nn.Embedding(
                config.ck_voc_dim, config.label_emb_dim
            )
            reinit_embedding_(self.label_embedding)

        self.in_dropout = CombinedDropout(*config.in_drop_rates)
        self.hid_dropout = CombinedDropout(*config.hid_drop_rates)

        if config.reduction.num_layers > 0 and config.reduction.out_dim > 0:
            self.reduction = config.reduction.instantiate()
            self.hid2logit = torch.nn.Linear(config.reduction.out_dim, config.voc_dim)
        else:
            self.hid2logit = torch.nn.Linear(config.reduction.in_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, "sigmoid")

        if config.ck_loss_weight > 0:
            self.ck_hid2logit = torch.nn.Linear(
                config.in_dim + config.size_emb_dim + config.label_emb_dim,
                config.ck_voc_dim,
            )
            reinit_layer_(self.ck_hid2logit, "sigmoid")

        self.criterion = config.instantiate_criterion(reduction="sum")

    def assign_chunks_pred(self, batch: Batch, batch_chunks_pred: List[List[tuple]]):
        """This method should be called on-the-fly for joint modeling."""
        for cs_obj, chunks_pred in zip(batch.cs_objs, batch_chunks_pred):
            if cs_obj.chunks_pred is None:
                cs_obj.chunks_pred = chunks_pred
                cs_obj.build(self)
                cs_obj.to(self.hid2logit.weight.device)

    def get_logits(self, batch: Batch, full_hidden: torch.Tensor):
        # full_hidden: (batch, step, hid_dim)
        batch_logits = []
        for i, cs_obj in enumerate(batch.cs_objs):
            num_chunks = len(cs_obj.chunks)
            if num_chunks == 0:
                logits = None
            else:
                # span_hidden: (num_chunks, span_size, hid_dim) -> (num_chunks, hid_dim)
                span_hidden = [
                    full_hidden[i, start:end] for label, start, end in cs_obj.chunks
                ]
                span_mask = seq_lens2mask(
                    torch.tensor(
                        [h.size(0) for h in span_hidden],
                        dtype=torch.long,
                        device=full_hidden.device,
                    )
                )
                span_hidden = torch.nn.utils.rnn.pad_sequence(
                    span_hidden, batch_first=True, padding_value=0.0
                )
                span_hidden = self.aggregating(
                    self.in_dropout(span_hidden), mask=span_mask
                )

                if hasattr(self, "size_embedding"):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cs_obj.span_size_ids)
                    span_hidden = torch.cat(
                        [span_hidden, self.in_dropout(size_embedded)], dim=-1
                    )

                if hasattr(self, "label_embedding"):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cs_obj.ck_label_ids)
                    span_hidden = torch.cat(
                        [span_hidden, self.in_dropout(label_embedded)], dim=-1
                    )

                if hasattr(self, "reduction"):
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

    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)

        losses = []
        for logits, cs_obj in zip(batch_logits, batch.cs_objs):
            if len(cs_obj.chunks) == 0:
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cs_obj.cs2label_id
                ck_label_ids_gold = cs_obj.ck_label_ids_gold
                if self.ck_loss_weight > 0:
                    logits, ck_logits = logits
                if hasattr(cs_obj, "non_mask"):
                    non_mask = cs_obj.non_mask
                    logits, label_ids = logits[non_mask], label_ids[non_mask]
                loss = self.criterion(logits, label_ids)
                if self.ck_loss_weight > 0:
                    loss = loss + self.ck_loss_weight * self.criterion(
                        ck_logits, ck_label_ids_gold
                    )
            losses.append(loss)
        return torch.stack(losses)

    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        batch_logits = self.get_logits(batch, full_hidden)

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
                    attributes = [
                        (label, chunk)
                        for label, chunk in zip(labels, cs_obj.chunks)
                        if label != self.none_label
                    ]
                    confidences = [
                        conf
                        for label, conf in zip(labels, confidences.cpu().tolist())
                        if label != self.none_label
                    ]
                else:
                    all_confidences = logits.sigmoid()
                    # Zero-out all entities according to <none> labels
                    all_confidences[
                        all_confidences[:, self.none_idx] > (1 - self.conf_thresh)
                    ] = 0
                    # Zero-out <none> labels for all entities
                    all_confidences[:, self.none_idx] = 0
                    assert all_confidences.size(0) == len(cs_obj.chunks)

                    all_confidences_list = all_confidences.cpu().tolist()
                    pos_entries = (
                        torch.nonzero(all_confidences > self.conf_thresh).cpu().tolist()
                    )

                    attributes = [
                        (self.idx2label[i], cs_obj.chunks[cidx])
                        for cidx, i in pos_entries
                    ]
                    confidences = [
                        all_confidences_list[cidx][i] for cidx, i in pos_entries
                    ]
                assert len(confidences) == len(attributes)
                attributes = self._filter(attributes)
            batch_attributes.append(attributes)

        return batch_attributes
