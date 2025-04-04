# -*- coding: utf-8 -*-
from collections import Counter
from typing import List

import torch

from ...nn.init import reinit_layer_
from ...nn.modules import CombinedDropout, SequenceAttention, SequencePooling
from ...wrapper import Batch
from .base import DecoderBase, DecoderMixinBase, SingleDecoderConfigBase


class TextClassificationDecoderMixin(DecoderMixinBase):
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

    def exemplify(self, data_entry: dict, training: bool = True):
        label = data_entry.get("label", None)
        return {
            "label_id": torch.tensor(
                self.label2idx[label] if label is not None else -100, dtype=torch.long
            )
        }

    def batchify(self, batch_examples: List[dict]):
        return {"label_ids": torch.stack([ex["label_id"] for ex in batch_examples])}

    def retrieve(self, batch: Batch):
        return [self.idx2label[label_id] for label_id in batch.label_ids.cpu().tolist()]

    def evaluate(self, y_gold: List[str], y_pred: List[str]):
        """Accuracy for text classification."""
        return sum(yp == yg for yp, yg in zip(y_gold, y_pred)) / len(y_gold)


class TextClassificationDecoderConfig(
    SingleDecoderConfigBase, TextClassificationDecoderMixin
):
    def __init__(self, **kwargs):
        self.in_drop_rates = kwargs.pop("in_drop_rates", (0.5, 0.0, 0.0))

        self.agg_mode = kwargs.pop("agg_mode", "multiplicative_attention")
        self.idx2label = kwargs.pop("idx2label", None)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name_sep.join([self.agg_mode, self.criterion])

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key)
            for key in ["in_dim", "in_drop_rates", "agg_mode", "criterion"]
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            counter.update([data_entry["label"] for data_entry in data])
        self.idx2label = list(counter.keys())

    def instantiate(self):
        return TextClassificationDecoder(self)


class TextClassificationDecoder(DecoderBase, TextClassificationDecoderMixin):
    def __init__(self, config: TextClassificationDecoderConfig):
        super().__init__()
        self.idx2label = config.idx2label

        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.hid2logit = torch.nn.Linear(config.in_dim, config.voc_dim)
        reinit_layer_(self.hid2logit, "sigmoid")

        if config.agg_mode.lower().endswith("_pooling"):
            self.aggregating = SequencePooling(
                mode=config.agg_mode.replace("_pooling", "")
            )
        elif config.agg_mode.lower().endswith("_attention"):
            self.aggregating = SequenceAttention(
                config.in_dim, scoring=config.agg_mode.replace("_attention", "")
            )

        self.criterion = config.instantiate_criterion(reduction="none")

    def forward(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.aggregating(self.dropout(full_hidden), mask=batch.mask)

        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        return self.criterion(logits, batch.label_ids)

    def decode(self, batch: Batch, full_hidden: torch.Tensor):
        # pooled_hidden: (batch, hid_dim)
        pooled_hidden = self.aggregating(full_hidden, mask=batch.mask)

        # logits: (batch, tag_dim)
        logits = self.hid2logit(pooled_hidden)
        return [
            self.idx2label[label_id]
            for label_id in logits.argmax(dim=-1).cpu().tolist()
        ]
