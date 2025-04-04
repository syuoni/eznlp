# -*- coding: utf-8 -*-
import copy
import logging
import math
from collections import Counter
from typing import List

import numpy
import torch

from ...nn.init import reinit_embedding_, reinit_layer_, reinit_vector_parameter_
from ...nn.modules import (
    BiAffineFusor,
    CombinedDropout,
    SoftLabelCrossEntropyLoss,
    TriAffineFusor,
)
from ...utils.chunk import chunk_pair_distance
from ...utils.relation import INV_REL_PREFIX
from ...wrapper import Batch
from ..encoder import EncoderConfig
from .base import DecoderBase, SingleDecoderConfigBase
from .boundaries import MAX_SIZE_ID_COV_RATE
from .chunks import MAX_DIST_ID_COV_RATE, ChunkPairs
from .span_rel_classification import ChunkPairsDecoderMixin

logger = logging.getLogger(__name__)


class MaskedSpanRelClsDecoderConfig(SingleDecoderConfigBase, ChunkPairsDecoderMixin):
    def __init__(self, **kwargs):
        self.use_context = kwargs.pop("use_context", True)
        self.context_mode = kwargs.pop("context_mode", "pair_specific")
        assert not (self.use_context and self.context_mode.lower().count("none"))
        # If `context_mode` is non-paired, `context_ext_win` should be strictly positive
        self.context_ext_win = kwargs.pop("context_ext_win", 0)
        self.context_exc_ck = kwargs.pop("context_exc_ck", True)

        self.size_emb_dim = kwargs.pop("size_emb_dim", 0)
        self.label_emb_dim = kwargs.pop("label_emb_dim", 0)
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
        if self.use_context:
            self.reduction_ctx = copy.deepcopy(self.reduction)
        if self.fusing_mode.lower().startswith("concat"):
            self.reduction_cat = copy.deepcopy(self.reduction)
            self.reduction_cat.hid_dim = self.reduction.hid_dim * self.num_hidden

        self.in_drop_rates = kwargs.pop("in_drop_rates", (0.4, 0.0, 0.0))
        self.hid_drop_rates = kwargs.pop("hid_drop_rates", (0.2, 0.0, 0.0))

        self.neg_sampling_rate = kwargs.pop("neg_sampling_rate", 1.0)

        self.ck_loss_weight = kwargs.pop("ck_loss_weight", 0)
        self.l2_loss_weight = kwargs.pop("l2_loss_weight", 0)

        self.sym_rel_labels = kwargs.pop("sym_rel_labels", [])
        self.comp_sym_rel = kwargs.pop("comp_sym_rel", False)
        self.use_inv_rel = kwargs.pop("use_inv_rel", False)
        self.check_rht_labels = kwargs.pop("check_rht_labels", False)
        self.none_label = kwargs.pop("none_label", "<none>")
        self.idx2label = kwargs.pop("idx2label", None)
        self.ck_none_label = kwargs.pop("ck_none_label", "<none>")
        self.idx2ck_label = kwargs.pop("idx2ck_label", None)

        # Simplified smoothing epsilon
        self.ss_epsilon = kwargs.pop("ss_epsilon", 0.0)
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
    def num_hidden(self):
        if self.use_context:
            return 3 if self.context_mode.lower().count("pair") else 4
        else:
            return 2

    @property
    def in_dim(self):
        return self._in_dim

    @in_dim.setter
    def in_dim(self, dim: int):
        if dim is not None:
            self._in_dim = dim
            self.reduction.in_dim = dim + self.size_emb_dim + self.label_emb_dim
            if self.use_context:
                self.reduction_ctx.in_dim = dim
            if self.fusing_mode.lower().startswith("concat"):
                self.reduction_cat.in_dim = (
                    self.in_dim * self.num_hidden
                    + self.size_emb_dim * 2
                    + self.label_emb_dim * 2
                )

    @property
    def criterion(self):
        if self.ss_epsilon > 0:
            return f"SS({self.ss_epsilon:.2f})"
        else:
            return super().criterion

    def instantiate_criterion(self, **kwargs):
        if self.criterion.lower().startswith("ss"):
            return SoftLabelCrossEntropyLoss(**kwargs)
        else:
            return super().instantiate_criterion(**kwargs)

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
            for label, head, tail in entry["relations"]
        )
        self.idx2label = [self.none_label] + list(counter.keys())
        if self.use_inv_rel:
            self.idx2label = self.idx2label + [
                f"{INV_REL_PREFIX}{label}"
                for label in self.idx2label
                if label != self.none_label and label not in self.sym_rel_labels
            ]

        counter = Counter(
            (label, head[0], tail[0])
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        )
        self.existing_rht_labels = set(list(counter.keys()))
        self.existing_ht_labels = set(rht[1:] for rht in self.existing_rht_labels)
        self.existing_self_rel = any(
            head[1:] == tail[1:]
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        )

        cp_distances = [
            chunk_pair_distance(head, tail)
            for data in partitions
            for entry in data
            for label, head, tail in entry["relations"]
        ]
        self.max_dist_id = math.ceil(numpy.quantile(cp_distances, MAX_DIST_ID_COV_RATE))
        self.max_cp_dist = max(cp_distances)

    def exemplify(self, entry: dict, training: bool = True):
        example = super().exemplify(entry, training)

        # Attention mask for each example
        cp_obj = example["cp_obj"]
        num_tokens, num_chunks = cp_obj.num_tokens, len(cp_obj.chunks)
        example["masked_span_bert_like"] = {
            "span_size_ids": cp_obj.span_size_ids,
            "cp_dist_ids": cp_obj.cp_dist_ids,
        }

        ck2tok_mask = torch.ones(num_chunks, num_tokens, dtype=torch.bool)
        for i, (label, start, end) in enumerate(cp_obj.chunks):
            for j in range(start, end):
                ck2tok_mask[i, j] = False
        example["masked_span_bert_like"].update({"ck2tok_mask": ck2tok_mask})

        if self.context_mode.lower().count("pair"):
            chunk_pairs = list(
                self.enumerate_chunk_pairs(cp_obj, return_valid_only=True)
            )
            num_pairs = len(chunk_pairs)
            pair2tok_mask = torch.ones(num_pairs, num_tokens, dtype=torch.bool)
            for i, ((_, h_start, h_end), (_, t_start, t_end)) in enumerate(chunk_pairs):
                for j in range(h_start, h_end):
                    pair2tok_mask[i, j] = False
                for j in range(t_start, t_end):
                    pair2tok_mask[i, j] = False

            ctx2tok_mask = torch.ones(num_pairs, num_tokens, dtype=torch.bool)
            for i, ((_, h_start, h_end), (_, t_start, t_end)) in enumerate(chunk_pairs):
                c_start = max(min(h_start, t_start) - self.context_ext_win, 0)
                c_end = min(max(h_end, t_end) + self.context_ext_win, num_tokens)
                for j in range(c_start, c_end):
                    # Context does not excluding chunk ranges, or `j` not in chunk ranges
                    if (not self.context_exc_ck) or pair2tok_mask[i, j].item():
                        ctx2tok_mask[i, j] = False

            example["masked_span_bert_like"].update(
                {"pair2tok_mask": pair2tok_mask, "ctx2tok_mask": ctx2tok_mask}
            )

        else:
            ctx2tok_mask = torch.ones(num_chunks, num_tokens, dtype=torch.bool)
            for i, (_, start, end) in enumerate(cp_obj.chunks):
                c_start = max(start - self.context_ext_win, 0)
                c_end = min(end + self.context_ext_win, num_tokens)
                for j in range(c_start, c_end):
                    if (not self.context_exc_ck) or ck2tok_mask[i, j].item():
                        ctx2tok_mask[i, j] = False

            example["masked_span_bert_like"].update({"ctx2tok_mask": ctx2tok_mask})

        return example

    def batchify(self, batch_examples: List[dict], batch_sub_mask: torch.Tensor):
        batch = super().batchify(batch_examples)

        # Batchify chunk-to-token attention mask
        # Remove `[CLS]` and `[SEP]`
        batch_sub_mask = batch_sub_mask[:, 2:]
        batch["masked_span_bert_like"] = {}
        for tensor_name in batch_examples[0]["masked_span_bert_like"].keys():
            if tensor_name.endswith("_mask"):
                max_num_items = max(
                    ex["masked_span_bert_like"][tensor_name].size(0)
                    for ex in batch_examples
                )
                batch_item2tok_mask = batch_sub_mask.unsqueeze(1).repeat(
                    1, max_num_items, 1
                )

                for i, ex in enumerate(batch_examples):
                    item2tok_mask = ex["masked_span_bert_like"][tensor_name]
                    num_items, num_tokens = item2tok_mask.size()
                    batch_item2tok_mask[i, :num_items, :num_tokens].logical_or_(
                        item2tok_mask
                    )

                batch["masked_span_bert_like"].update(
                    {tensor_name: batch_item2tok_mask}
                )

            elif tensor_name.endswith("_ids"):
                batch_ids = [
                    ex["masked_span_bert_like"][tensor_name] for ex in batch_examples
                ]
                batch_ids = torch.nn.utils.rnn.pad_sequence(
                    batch_ids, batch_first=True, padding_value=0
                )
                batch["masked_span_bert_like"].update({tensor_name: batch_ids})

        return batch

    def instantiate(self):
        return MaskedSpanRelClsDecoder(self)


class MaskedSpanRelClsDecoder(DecoderBase, ChunkPairsDecoderMixin):
    def __init__(self, config: MaskedSpanRelClsDecoderConfig):
        super().__init__()
        self.use_context = config.use_context
        self.context_mode = config.context_mode
        self.fusing_mode = config.fusing_mode
        self.max_size_id = config.max_size_id
        self.neg_sampling_rate = config.neg_sampling_rate
        self.ck_loss_weight = config.ck_loss_weight
        self.l2_loss_weight = config.l2_loss_weight
        self.sym_rel_labels = config.sym_rel_labels
        self.comp_sym_rel = config.comp_sym_rel
        self.use_inv_rel = config.use_inv_rel
        self.check_rht_labels = config.check_rht_labels
        self.none_label = config.none_label
        self.idx2label = config.idx2label
        self.ck_none_label = config.ck_none_label
        self.idx2ck_label = config.idx2ck_label
        self.ss_epsilon = config.ss_epsilon
        self.existing_rht_labels = config.existing_rht_labels
        self.existing_ht_labels = config.existing_ht_labels
        self.existing_self_rel = config.existing_self_rel
        self.max_cp_dist = config.max_cp_dist
        self.max_dist_id = config.max_dist_id

        if config.use_context:
            # Trainable context vector for overlapping chunk pairs
            self.zero_context = torch.nn.Parameter(torch.empty(config.in_dim))
            reinit_vector_parameter_(self.zero_context)
            # A placeholder context vector for invalid chunk pairs
            self.register_buffer("none_context", torch.zeros(config.in_dim))

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

        if config.fusing_mode.lower().startswith("concat"):
            if config.reduction_cat.num_layers > 0 and config.reduction_cat.out_dim > 0:
                self.reduction_cat = config.reduction_cat.instantiate()
                self.hid2logit = torch.nn.Linear(
                    config.reduction_cat.out_dim, config.voc_dim
                )
            else:
                self.hid2logit = torch.nn.Linear(
                    config.reduction_cat.in_dim, config.voc_dim
                )
            reinit_layer_(self.hid2logit, "sigmoid")

        elif config.fusing_mode.lower().startswith("affine"):
            self.reduction_head = config.reduction.instantiate()
            self.reduction_tail = config.reduction.instantiate()
            if config.use_context:
                if self.context_mode.lower().count("pair"):
                    self.reduction_ctx = config.reduction_ctx.instantiate()
                else:
                    self.reduction_hctx = config.reduction_ctx.instantiate()
                    self.reduction_tctx = config.reduction_ctx.instantiate()
                    self.ctx_fusor = BiAffineFusor(
                        config.reduction.out_dim, config.reduction.out_dim
                    )
                self.hid2logit = TriAffineFusor(
                    config.reduction.out_dim, config.voc_dim
                )
            else:
                self.hid2logit = BiAffineFusor(config.reduction.out_dim, config.voc_dim)

        if config.ck_loss_weight > 0:
            self.ck_hid2logit = torch.nn.Linear(
                config.in_dim + config.size_emb_dim + config.label_emb_dim,
                config.ck_voc_dim,
            )
            reinit_layer_(self.ck_hid2logit, "sigmoid")

        self.criterion = config.instantiate_criterion(reduction="sum")

    def _collect_pair_specific_contexts(
        self, ctx_query_hidden: torch.Tensor, i: int, cp_obj: ChunkPairs
    ):
        # contexts: (num_chunks^2, hid_dim) -> (num_chunks, num_chunks, hid_dim)
        contexts = []
        valid_pair_idx = 0
        for (
            (_, h_start, h_end),
            (_, t_start, t_end),
            is_valid,
        ) in self.enumerate_chunk_pairs(cp_obj):
            if not is_valid:
                contexts.append(self.none_context)
            else:
                contexts.append(ctx_query_hidden[i, valid_pair_idx])
                valid_pair_idx += 1
        contexts = self.in_dropout(torch.stack(contexts))
        contexts = contexts.view(len(cp_obj.chunks), len(cp_obj.chunks), -1)
        return contexts

    def get_logits(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        span_query_hidden: torch.Tensor,
        ctx_query_hidden: torch.Tensor,
    ):
        # full_hidden: (batch, step, hid_dim)
        # span_query_hidden/ctx_query_hidden: (batch, num_chunks, hid_dim)
        batch_logits = []
        for i, cp_obj in enumerate(batch.cp_objs):
            if not cp_obj.has_valid_cp:
                logits = None
            else:
                # span_hidden: (num_chunks, hid_dim)
                span_hidden = span_query_hidden[i, : len(cp_obj.chunks)]

                if hasattr(self, "size_embedding"):
                    # size_embedded: (num_chunks, emb_dim)
                    size_embedded = self.size_embedding(cp_obj.span_size_ids)
                    span_hidden = torch.cat([span_hidden, size_embedded], dim=-1)

                if hasattr(self, "label_embedding"):
                    # label_embedded: (num_chunks, emb_dim)
                    label_embedded = self.label_embedding(cp_obj.ck_label_ids)
                    span_hidden = torch.cat([span_hidden, label_embedded], dim=-1)

                head_hidden = (
                    self.in_dropout(span_hidden)
                    .unsqueeze(1)
                    .expand(-1, len(cp_obj.chunks), -1)
                )
                tail_hidden = (
                    self.in_dropout(span_hidden)
                    .unsqueeze(0)
                    .expand(len(cp_obj.chunks), -1, -1)
                )

                if self.use_context:
                    if self.context_mode.lower().count("pair"):
                        # contexts: (num_chunks, num_chunks, hid_dim)
                        contexts = self._collect_pair_specific_contexts(
                            ctx_query_hidden, i, cp_obj
                        )
                    else:
                        # ctx_hidden: (num_chunks, hid_dim) -> (num_chunks, num_chunks, hid_dim)
                        ctx_hidden = ctx_query_hidden[i, : len(cp_obj.chunks)]
                        h_contexts = (
                            self.in_dropout(ctx_hidden)
                            .unsqueeze(1)
                            .expand(-1, len(cp_obj.chunks), -1)
                        )
                        t_contexts = (
                            self.in_dropout(ctx_hidden)
                            .unsqueeze(0)
                            .expand(len(cp_obj.chunks), -1, -1)
                        )

                if self.fusing_mode.startswith("concat"):
                    # hidden_cat: (num_chunks, num_chunks, hid_dim*4)
                    hidden_cat = torch.cat([head_hidden, tail_hidden], dim=-1)
                    if self.use_context:
                        if self.context_mode.lower().count("pair"):
                            hidden_cat = torch.cat([hidden_cat, contexts], dim=-1)
                        else:
                            hidden_cat = torch.cat(
                                [hidden_cat, h_contexts, t_contexts], dim=-1
                            )
                    if hasattr(self, "reduction_cat"):
                        reduced_cat = self.reduction_cat(hidden_cat)
                        logits = self.hid2logit(self.hid_dropout(reduced_cat))
                    else:
                        # logits: (num_chunks, num_chunks, logit_dim)
                        logits = self.hid2logit(hidden_cat)

                elif self.fusing_mode.lower().startswith("affine"):
                    reduced_head = self.reduction_head(head_hidden)
                    reduced_tail = self.reduction_tail(tail_hidden)
                    if self.use_context:
                        if self.context_mode.lower().count("pair"):
                            reduced_ctx = self.reduction_ctx(contexts)
                        else:
                            reduced_hctx = self.reduction_hctx(h_contexts)
                            reduced_tctx = self.reduction_tctx(t_contexts)
                            reduced_ctx = self.ctx_fusor(
                                self.hid_dropout(reduced_hctx),
                                self.hid_dropout(reduced_tctx),
                            )
                        logits = self.hid2logit(
                            self.hid_dropout(reduced_head),
                            self.hid_dropout(reduced_tail),
                            self.hid_dropout(reduced_ctx),
                        )
                    else:
                        logits = self.hid2logit(
                            self.hid_dropout(reduced_head),
                            self.hid_dropout(reduced_tail),
                        )

                if self.ck_loss_weight > 0:
                    ck_logits = self.ck_hid2logit(span_hidden)
                    logits = (logits, ck_logits)

            batch_logits.append(logits)

        return batch_logits

    def forward(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        span_query_hidden: torch.Tensor,
        ctx_query_hidden: torch.Tensor,
    ):
        batch_logits = self.get_logits(
            batch, full_hidden, span_query_hidden, ctx_query_hidden
        )

        losses = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if not cp_obj.has_valid_cp:
                loss = torch.tensor(0.0, device=full_hidden.device)
            else:
                label_ids = cp_obj.cp2label_id
                ck_label_ids_gold = cp_obj.ck_label_ids_gold
                if self.ck_loss_weight > 0:
                    logits, ck_logits = logits
                logits, label_ids = logits[cp_obj.non_mask], label_ids[cp_obj.non_mask]
                loss = self.criterion(logits, label_ids)
                if self.ck_loss_weight > 0:
                    loss = loss + self.ck_loss_weight * self.criterion(
                        ck_logits, ck_label_ids_gold
                    )
                if self.l2_loss_weight > 0 and self.fusing_mode.lower().startswith(
                    "affine"
                ):
                    loss = loss + self.l2_loss_weight * (self.hid2logit.U**2).sum()
            losses.append(loss)
        return torch.stack(losses)

    def decode(
        self,
        batch: Batch,
        full_hidden: torch.Tensor,
        span_query_hidden: torch.Tensor,
        ctx_query_hidden: torch.Tensor,
    ):
        batch_logits = self.get_logits(
            batch, full_hidden, span_query_hidden, ctx_query_hidden
        )

        batch_relations = []
        for logits, cp_obj in zip(batch_logits, batch.cp_objs):
            if not cp_obj.has_valid_cp:
                relations = []
            else:
                if self.ck_loss_weight > 0:
                    logits, _ = logits
                confidences, label_ids = logits.softmax(dim=-1).max(dim=-1)
                labels = [self.idx2label[i] for i in label_ids.flatten().cpu().tolist()]
                relations = [
                    (label, head, tail)
                    for label, (head, tail, is_valid) in zip(
                        labels, self.enumerate_chunk_pairs(cp_obj)
                    )
                    if is_valid and label != self.none_label
                ]
                relations = self._filter(relations)
            batch_relations.append(relations)

        return batch_relations
