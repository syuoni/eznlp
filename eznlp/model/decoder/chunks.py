# -*- coding: utf-8 -*-
import random
from typing import List, Union

import torch

from ...utils.chunk import chunk_pair_distance
from ...utils.relation import detect_inverse
from ...wrapper import TargetWrapper
from .base import DecoderBase, SingleDecoderConfigBase

MAX_DIST_ID_COV_RATE = 0.975
MAX_CP_DIST_TOL = 5


class ChunkPairs(TargetWrapper):
    """A wrapper of chunk-pairs with underlying relations.
    This object enumerates all pairs between all positive spans (entity spans).

    For pipeline modeling, `chunks_pred` is pre-computed, and thus initially non-empty in `entry`;
    For joint modeling, `chunks_pred` is computed on-the-fly, and thus initially empty in `entry`.

    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence,
         'chunks': List[tuple],
         'relations': List[tuple]}
    """

    def __init__(
        self, entry: dict, config: SingleDecoderConfigBase, training: bool = True
    ):
        super().__init__(training)

        # `max_span_size` is the threshold to filter out extra-length chunks; `None` means no filtering
        self.max_span_size = getattr(config, "max_span_size", None)
        self.num_tokens = len(entry["tokens"])
        self.chunks_gold = entry.get("chunks", None)
        self.chunks_pred = entry.get("chunks_pred", None)
        self.relations = entry.get("relations", None)
        if "tok2sent_idx" in entry:
            self.tok2sent_idx = entry["tok2sent_idx"]
        if self.chunks_pred is not None:
            self.build(config)

    @property
    def chunks_pred(self):
        return self._chunks_pred

    @chunks_pred.setter
    def chunks_pred(self, chunks: List[tuple]):
        # `chunks_pred` is unchangable once set
        # In the evaluation phase, the outside chunk-decoder should produce deterministic predicted chunks
        assert getattr(self, "_chunks_pred", None) is None
        self._chunks_pred = chunks

        if self.chunks_pred is not None:
            # Gold chunks are *inaccessible* in the evaluation mode
            if self.training and self.chunks_gold is not None:
                self.chunks = [ck for ck in self.chunks_gold]
            else:
                self.chunks = []

            # Do not use ```chunks = list(set(chunks + chunks_pred))```, which may return non-deterministic order.
            # In the early stage of joint training, the chunk-decoder may produce too many predicted chunks, so do sampling here.
            existing_spans = {ck[1:] for ck in self.chunks}
            chunks_extra = [
                ck for ck in self.chunks_pred if ck[1:] not in existing_spans
            ]
            num_neg_chunks = max(int(self.num_tokens * 0.35), 20)
            if len(chunks_extra) > num_neg_chunks:
                chunks_extra = random.sample(chunks_extra, num_neg_chunks)
            self.chunks.extend(chunks_extra)

            if self.max_span_size is not None:
                self.chunks = [
                    ck for ck in self.chunks if ck[2] - ck[1] <= self.max_span_size
                ]

            self.chunk2idx = {ck: i for i, ck in enumerate(self.chunks)}

    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_chunks = len(self.chunks)

        self.span_size_ids = torch.tensor(
            [end - start - 1 for label, start, end in self.chunks], dtype=torch.long
        )
        self.span_size_ids.masked_fill_(
            self.span_size_ids > config.max_size_id, config.max_size_id
        )

        # `ck_label_ids` is for model input; it includes predicted chunk labels which may be incorrect
        # `ck_label_ids_gold` is for model target; it's chunk labels are all consistent with ground truth
        self.ck_label_ids = torch.tensor(
            [config.ck_label2idx[label] for label, start, end in self.chunks],
            dtype=torch.long,
        )
        span2ck_label_gold = (
            {}
            if self.chunks_gold is None
            else {(start, end): label for label, start, end in self.chunks_gold}
        )
        self.ck_label_ids_gold = torch.tensor(
            [
                config.ck_label2idx[
                    span2ck_label_gold.get((start, end), config.ck_none_label)
                ]
                for label, start, end in self.chunks
            ],
            dtype=torch.long,
        )

        self.cp_dist_ids = torch.tensor(
            [
                chunk_pair_distance(head, tail)
                for head, tail in config.enumerate_chunk_pairs(
                    self, return_valid_only=True
                )
            ],
            dtype=torch.long,
        )
        self.cp_dist_ids.masked_fill_(self.cp_dist_ids < 0, 0)
        self.cp_dist_ids.masked_fill_(
            self.cp_dist_ids > config.max_dist_id, config.max_dist_id
        )

        is_valid_list = [
            is_valid for _, _, is_valid in config.enumerate_chunk_pairs(self)
        ]
        self.has_valid_cp = any(is_valid_list)
        self.non_mask = torch.tensor(is_valid_list, dtype=torch.bool).view(
            num_chunks, num_chunks
        )

        if self.training and config.neg_sampling_rate < 1:
            non_mask_rate = config.neg_sampling_rate * torch.ones(
                num_chunks, num_chunks, dtype=torch.float
            )
            for label, head, tail in self.relations:
                if head in self.chunk2idx and tail in self.chunk2idx:
                    hk = self.chunk2idx[head]
                    tk = self.chunk2idx[tail]
                    non_mask_rate[hk, tk] = 1

            # Bernoulli sampling according probability in `non_mask_rate`
            non_mask_sampling = non_mask_rate.bernoulli().bool()
            self.non_mask.logical_and_(non_mask_sampling)

        if self.relations is not None:
            self.cp2label_id = torch.full(
                (num_chunks, num_chunks), config.none_idx, dtype=torch.long
            )
            inverse_relations = (
                detect_inverse(self.relations) if config.use_inv_rel else []
            )
            for label, head, tail in self.relations + inverse_relations:
                if head in self.chunk2idx and tail in self.chunk2idx:
                    hk = self.chunk2idx[head]
                    tk = self.chunk2idx[tail]
                    self.cp2label_id[hk, tk] = config.label2idx[label]
                else:
                    # `head`/`tail` may not appear in `chunks` in case of:
                    # (1) in the evaluation phase where `chunks_gold` are not allowed to access.
                    # In this case, `cp2label_id` is only for forwarding to a "fake" loss, but not for backwarding.
                    # (2) `head`/`tail` is filtered out because of exceeding `max_span_size`.
                    assert (
                        (not self.training)
                        or (head[2] - head[1] > self.max_span_size)
                        or (tail[2] - tail[1] > self.max_span_size)
                    )

            if config.ss_epsilon > 0:
                # Soft label loss for simplified smoothing
                self.cp2label_id = torch.nn.functional.one_hot(
                    self.cp2label_id, num_classes=config.voc_dim
                ).float()
                self.cp2label_id = self.cp2label_id * (1 - config.ss_epsilon)
                self.cp2label_id[:, :, config.none_idx] += 1 - self.cp2label_id.sum(
                    dim=-1
                )

                self.ck_label_ids_gold = torch.nn.functional.one_hot(
                    self.ck_label_ids_gold, num_classes=config.ck_voc_dim
                ).float()
                self.ck_label_ids_gold = self.ck_label_ids_gold * (
                    1 - config.ss_epsilon
                )
                self.ck_label_ids_gold[
                    :, config.ck_none_idx
                ] += 1 - self.ck_label_ids_gold.sum(dim=-1)


class ChunkSingles(TargetWrapper):
    """A wrapper of chunk-singles with underlying attributes.
    This object enumerates all positive spans (entity spans).

    For pipeline modeling, `chunks_pred` is pre-computed, and thus initially non-empty in `entry`;
    For joint modeling, `chunks_pred` is computed on-the-fly, and thus initially empty in `entry`.

    Parameters
    ----------
    entry: dict
        {'tokens': TokenSequence,
         'chunks': List[tuple],
         'attributes': List[tuple]}
    """

    def __init__(
        self, entry: dict, config: SingleDecoderConfigBase, training: bool = True
    ):
        super().__init__(training)

        # `max_span_size` is the threshold to filter out extra-length chunks; `None` means no filtering
        self.max_span_size = getattr(config, "max_span_size", None)
        self.num_tokens = len(entry["tokens"])
        self.chunks_gold = entry.get("chunks", None)
        self.chunks_pred = entry.get("chunks_pred", None)
        self.attributes = entry.get("attributes", None)
        if self.chunks_pred is not None:
            self.build(config)

    @property
    def chunks_pred(self):
        return self._chunks_pred

    @chunks_pred.setter
    def chunks_pred(self, chunks: List[tuple]):
        # `chunks_pred` is unchangable once set
        # In the evaluation phase, the outside chunk-decoder should produce deterministic predicted chunks
        assert getattr(self, "_chunks_pred", None) is None
        self._chunks_pred = chunks

        if self.chunks_pred is not None:
            # Gold chunks are *inaccessible* in the evaluation mode
            if self.training and self.chunks_gold is not None:
                self.chunks = [ck for ck in self.chunks_gold]
            else:
                self.chunks = []

            # Do not use ```chunks = list(set(chunks + chunks_pred))```, which may return non-deterministic order.
            # In the early stage of joint training, the chunk-decoder may produce too many predicted chunks, so do sampling here.
            existing_spans = {ck[1:] for ck in self.chunks}
            chunks_extra = [
                ck for ck in self.chunks_pred if ck[1:] not in existing_spans
            ]
            num_neg_chunks = max(int(self.num_tokens * 0.35), 20)
            if len(chunks_extra) > num_neg_chunks:
                chunks_extra = random.sample(chunks_extra, num_neg_chunks)
            self.chunks.extend(chunks_extra)

            if self.max_span_size is not None:
                self.chunks = [
                    ck for ck in self.chunks if ck[2] - ck[1] <= self.max_span_size
                ]

            self.chunk2idx = {ck: i for i, ck in enumerate(self.chunks)}

    def build(self, config: Union[SingleDecoderConfigBase, DecoderBase]):
        num_chunks = len(self.chunks)

        self.span_size_ids = torch.tensor(
            [end - start - 1 for label, start, end in self.chunks], dtype=torch.long
        )
        self.span_size_ids.masked_fill_(
            self.span_size_ids > config.max_size_id, config.max_size_id
        )

        # `ck_label_ids` is for model input; it includes predicted chunk labels which may be incorrect
        # `ck_label_ids_gold` is for model target; it's chunk labels are all consistent with ground truth
        self.ck_label_ids = torch.tensor(
            [config.ck_label2idx[label] for label, start, end in self.chunks],
            dtype=torch.long,
        )
        span2ck_label_gold = (
            {}
            if self.chunks_gold is None
            else {(start, end): label for label, start, end in self.chunks_gold}
        )
        self.ck_label_ids_gold = torch.tensor(
            [
                config.ck_label2idx[
                    span2ck_label_gold.get((start, end), config.ck_none_label)
                ]
                for label, start, end in self.chunks
            ],
            dtype=torch.long,
        )

        if self.training and config.neg_sampling_rate < 1:
            non_mask_rate = config.neg_sampling_rate * torch.ones(
                num_chunks, dtype=torch.float
            )
            for label, chunk in self.attributes:
                if chunk in self.chunk2idx:
                    k = self.chunk2idx[chunk]
                    non_mask_rate[k] = 1

            # Bernoulli sampling according probability in `non_mask_rate`
            self.non_mask = non_mask_rate.bernoulli().bool()

        if self.attributes is not None:
            if not config.multilabel:
                self.cs2label_id = torch.full(
                    (num_chunks,), config.none_idx, dtype=torch.long
                )
                for label, chunk in self.attributes:
                    if chunk in self.chunk2idx:
                        k = self.chunk2idx[chunk]
                        self.cs2label_id[k] = config.label2idx[label]
                    else:
                        assert (not self.training) or (
                            chunk[2] - chunk[1] > self.max_span_size
                        )
            else:
                # `torch.nn.BCEWithLogitsLoss` uses float tensor as target
                self.cs2label_id = torch.zeros(
                    num_chunks, config.voc_dim, dtype=torch.float
                )
                for label, chunk in self.attributes:
                    if chunk in self.chunk2idx:
                        k = self.chunk2idx[chunk]
                        self.cs2label_id[k, config.label2idx[label]] = 1
                    else:
                        # `head`/`tail` may not appear in `chunks` in case of:
                        # (1) in the evaluation phase where `chunks_gold` are not allowed to access.
                        # In this case, `cp2label_id` is only for forwarding to a "fake" loss, but not for backwarding.
                        # (2) `head`/`tail` is filtered out because of exceeding `max_span_size`.
                        assert (not self.training) or (
                            chunk[2] - chunk[1] > self.max_span_size
                        )

                # Assign `<none>` label
                self.cs2label_id[:, config.none_idx] = (self.cs2label_id == 0).all(
                    dim=1
                )

                # One-hot labels for BCE loss
                self.ck_label_ids_gold = torch.nn.functional.one_hot(
                    self.ck_label_ids_gold, num_classes=config.ck_voc_dim
                ).float()
