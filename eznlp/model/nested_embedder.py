# -*- coding: utf-8 -*-
from collections import Counter
from typing import List

import torch

from ..nn.functional import seq_lens2mask
from ..nn.modules import SequencePooling
from ..token import TokenSequence
from ..vocab import Vocab
from .embedder import OneHotConfig, OneHotEmbedder
from .encoder import EncoderConfig


class NestedOneHotConfig(OneHotConfig):
    """Config of an embedder, designed for features with the structure: `step * channel * inner_step`.
    At each position exists one (or multiple) sequence.

    If multiple sequences exist, they share a common vocabulary and corresponding embedding layer.
    """

    def __init__(self, **kwargs):
        self.tokens_key = kwargs.pop("tokens_key", "tokens")
        self.num_channels = kwargs.pop("num_channels", 1)
        self.squeeze = kwargs.pop("squeeze", True)
        if self.squeeze:
            assert self.num_channels == 1

        self.encoder: EncoderConfig = kwargs.pop("encoder", None)
        self.agg_mode = kwargs.pop("agg_mode", "mean_pooling")

        super().__init__(**kwargs)
        if self.encoder is not None:
            self.encoder.in_dim = self.emb_dim

    @property
    def valid(self):
        if self.encoder is not None and not self.encoder.valid:
            return False
        return all(
            attr is not None
            for name, attr in self.__dict__.items()
            if name not in ("vectors", "encoder")
        )

    @property
    def out_dim(self):
        if self.encoder is not None:
            return self.encoder.out_dim * self.num_channels
        else:
            return self.emb_dim * self.num_channels

    def _inner_sequences(self, tokens: TokenSequence):
        for tok_field in getattr(tokens, self.field):
            if self.squeeze:
                yield tok_field
            else:
                yield from tok_field

    def build_vocab(self, *partitions):
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry[self.tokens_key]):
                    counter.update(inner_seq)
                    if self.max_len is None or len(inner_seq) > self.max_len:
                        self.max_len = len(inner_seq)

        self.vocab = Vocab(
            counter, min_freq=self.min_freq, specials=self.specials, specials_first=True
        )

    def exemplify(self, tokens: TokenSequence):
        inner_ids_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_ids_list.append(torch.tensor([self.vocab[x] for x in inner_seq]))

        # inner_ids: (step*num_channels, inner_step)
        return {"inner_ids": inner_ids_list}

    def batchify(self, batch_ex: List[dict]):
        batch_inner_ids = [
            inner_ids for ex in batch_ex for inner_ids in ex["inner_ids"]
        ]
        inner_seq_lens = torch.tensor(
            [inner_ids.size(0) for inner_ids in batch_inner_ids]
        )
        inner_mask = seq_lens2mask(inner_seq_lens)
        batch_inner_ids = torch.nn.utils.rnn.pad_sequence(
            batch_inner_ids, batch_first=True, padding_value=self.pad_idx
        )
        # inner_ids: (batch*step*num_channels, inner_step)
        return {"inner_ids": batch_inner_ids, "inner_mask": inner_mask}

    def instantiate(self):
        return NestedOneHotEmbedder(self)


class NestedOneHotEmbedder(OneHotEmbedder):
    def __init__(self, config: NestedOneHotConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        if config.encoder is not None:
            self.encoder = config.encoder.instantiate()

        self.agg_mode = config.agg_mode
        if self.agg_mode.lower().endswith(
            "_pooling"
        ) or self.agg_mode.lower().startswith("rnn_last"):
            self.aggregating = SequencePooling(
                mode=self.agg_mode.replace("_pooling", "")
            )

    def _restore_outer_shapes(self, x: torch.Tensor, seq_lens: torch.LongTensor):
        offsets = [0] + (seq_lens * self.num_channels).cumsum(dim=0).cpu().tolist()
        # x: (batch, step*num_channels, emb_dim/hid_dim)
        x = torch.nn.utils.rnn.pad_sequence(
            [x[s:e] for s, e in zip(offsets[:-1], offsets[1:])],
            batch_first=True,
            padding_value=0.0,
        )
        # x: (batch, step, num_channels * emb_dim/hid_dim)
        return x.view(x.size(0), -1, self.num_channels * x.size(-1))

    def forward(
        self,
        inner_ids: torch.LongTensor,
        inner_mask: torch.BoolTensor,
        seq_lens: torch.LongTensor,
        inner_weight: torch.FloatTensor = None,
    ):
        assert (seq_lens * self.num_channels).sum().item() == inner_ids.size(0)

        # embedded: (batch*step*num_channels, inner_step, emb_dim)
        embedded = self.embedding(inner_ids)

        # TODO: positional embedding?

        # encoding -> aggregating
        # hidden: (batch*step*num_channels, inner_step, hid_dim)
        # agg_hidden: (batch*step*num_channels, hid_dim)
        if hasattr(self, "encoder"):
            hidden = self.encoder(embedded, inner_mask)
            agg_hidden = self.aggregating(hidden, inner_mask, weight=inner_weight)
        else:
            agg_hidden = self.aggregating(embedded, inner_mask, weight=inner_weight)

        # Restore outer shapes (token-level steps)
        return self._restore_outer_shapes(agg_hidden, seq_lens)


class SoftLexiconConfig(NestedOneHotConfig):
    """Config of a soft lexicon embedder.

    References
    ----------
    Ma et al. (2020). Simplify the usage of lexicon in Chinese NER. ACL 2020.
    """

    def __init__(self, **kwargs):
        kwargs["field"] = kwargs.pop("field", "softlexicon")
        kwargs["num_channels"] = kwargs.pop("num_channels", 4)
        kwargs["squeeze"] = kwargs.pop("squeeze", False)

        kwargs["emb_dim"] = kwargs.pop("emb_dim", 50)
        kwargs["agg_mode"] = kwargs.pop("agg_mode", "wtd_mean_pooling")
        super().__init__(**kwargs)

    def __repr__(self):
        repr_attr_dict = {
            key: getattr(self, key) for key in self.__dict__.keys() if key != "freqs"
        }
        return self._repr_non_config_attrs(repr_attr_dict)

    def build_freqs(self, *partitions):
        """Ma et al. (2020): The statistical data set is constructed from a combination
        of *training* and *developing* data of the task.
        In addition, note that the frequency of `w` does not increase if `w` is
        covered by another sub-sequence that matches the lexicon
        """
        counter = Counter()
        for data in partitions:
            for data_entry in data:
                for inner_seq in self._inner_sequences(data_entry[self.tokens_key]):
                    counter.update(inner_seq)

        # NOTE: Set the minimum frequecy as 1, to avoid OOV tokens being ignored
        self.freqs = {tok: 1 for tok in self.vocab.itos}
        self.freqs.update(counter)
        self.freqs["<pad>"] = 0

    def exemplify(self, tokens: TokenSequence):
        example = super().exemplify(tokens)

        inner_freqs_list = []
        for inner_seq in self._inner_sequences(tokens):
            inner_freqs_list.append(torch.tensor([self.freqs[x] for x in inner_seq]))

        example["inner_freqs"] = inner_freqs_list
        return example

    def batchify(self, batch_ex: List[dict]):
        batch = super().batchify(batch_ex)

        batch_inner_freqs = [
            inner_freqs for ex in batch_ex for inner_freqs in ex["inner_freqs"]
        ]
        batch_inner_freqs = torch.nn.utils.rnn.pad_sequence(
            batch_inner_freqs, batch_first=True, padding_value=0
        )
        batch["inner_weight"] = batch_inner_freqs
        return batch


class CharConfig(NestedOneHotConfig):
    """Config of a character embedder."""

    def __init__(self, **kwargs):
        kwargs["field"] = kwargs.pop("field", "raw_text")
        kwargs["num_channels"] = kwargs.pop("num_channels", 1)
        kwargs["squeeze"] = kwargs.pop("squeeze", True)

        kwargs["emb_dim"] = kwargs.pop("emb_dim", 16)
        kwargs["encoder"] = kwargs.pop(
            "encoder",
            EncoderConfig(
                arch="LSTM", hid_dim=128, num_layers=1, in_drop_rates=(0.5, 0.0, 0.0)
            ),
        )
        if kwargs["encoder"].arch.lower() in ("lstm", "gru"):
            kwargs["agg_mode"] = kwargs.pop("agg_mode", "rnn_last")
        elif kwargs["encoder"].arch.lower() in ("conv", "gehring"):
            kwargs["agg_mode"] = kwargs.pop("agg_mode", "max_pooling")
        super().__init__(**kwargs)

    @property
    def name(self):
        return f"Char{self.encoder.arch}"
