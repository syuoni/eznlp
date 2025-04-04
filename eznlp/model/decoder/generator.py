# -*- coding: utf-8 -*-
import itertools
from typing import List

import nltk
import torch

from ...nn.init import (
    reinit_gru_,
    reinit_layer_,
    reinit_lstm_,
    reinit_vector_parameter_,
)
from ...nn.modules import (
    CombinedDropout,
    ConvBlock,
    SequenceAttention,
    SequencePooling,
    TransformerDecoderBlock,
)
from ...wrapper import Batch
from ..embedder import OneHotConfig, VocabMixin
from .base import DecoderBase, DecoderMixinBase, SingleDecoderConfigBase


class GeneratorMixin(DecoderMixinBase, VocabMixin):
    def exemplify(self, entry: dict, training: bool = True):
        example = {}

        if training:
            # Single ground-truth sentence for training
            example["trg_tok_ids"] = self.embedding.exemplify(entry["trg_tokens"])
        else:
            assert "trg_tokens" not in entry
            # Notes: The padding positions are ignored in loss computation, so the dev. loss will always be 0
            example["trg_tok_ids"] = torch.tensor(
                [self.embedding.sos_idx]
                + [self.embedding.pad_idx] * (self.max_len - 1),
                dtype=torch.long,
            )

        if "full_trg_tokens" in entry:
            # Multiple reference sentences for evaluation
            example["full_trg_tokenized_text"] = [
                tokens.text for tokens in entry["full_trg_tokens"]
            ]

        return example

    def batchify(self, batch_examples: List[dict]):
        batch = {}
        batch["trg_tok_ids"] = self.embedding.batchify(
            [ex["trg_tok_ids"] for ex in batch_examples]
        )

        if "full_trg_tokenized_text" in batch_examples[0]:
            batch["full_trg_tokenized_text"] = [
                ex["full_trg_tokenized_text"] for ex in batch_examples
            ]

        return batch

    def retrieve(self, batch: Batch):
        return batch.full_trg_tokenized_text

    def evaluate(self, y_gold: List[List[List[str]]], y_pred: List[List[str]]):
        assert (
            isinstance(y_gold[0], list)
            and isinstance(y_gold[0][0], list)
            and isinstance(y_gold[0][0][0], str)
        )
        assert isinstance(y_pred[0], list) and (
            len(y_pred[0]) == 0 or isinstance(y_pred[0][0], str)
        )
        # torchtext.data.metrics.bleu_score(candidate_corpus=y_pred, references_corpus=y_gold)
        return nltk.translate.bleu_score.corpus_bleu(
            list_of_references=y_gold, hypotheses=y_pred
        )


class GeneratorConfig(SingleDecoderConfigBase, GeneratorMixin):
    def __init__(self, **kwargs):
        self.arch = kwargs.pop("arch", "LSTM")
        self.hid_dim = kwargs.pop("hid_dim", 128)

        # Attention is the default structure
        if self.arch.lower() in ("lstm", "gru"):
            self.init_ctx_mode = kwargs.pop("init_ctx_mode", "mean_pooling")
            self.embedding = kwargs.pop(
                "embedding",
                OneHotConfig(
                    tokens_key="trg_tokens", field="text", has_sos=True, has_eos=True
                ),
            )
            self.num_layers = kwargs.pop("num_layers", 1)
            self.num_heads = kwargs.pop("num_heads", 1)
            self.scoring = kwargs.pop("scoring", "additive")
            self.in_drop_rates = kwargs.pop("in_drop_rates", (0.5, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.5)
            self.shortcut = kwargs.pop("shortcut", True)

        elif self.arch.lower() in ("conv", "gehring"):
            self.kernel_size = kwargs.pop("kernel_size", 3)
            self.scale = kwargs.pop("scale", 0.5**0.5)
            self.embedding = kwargs.pop(
                "embedding",
                OneHotConfig(
                    tokens_key="trg_tokens",
                    field="text",
                    has_sos=True,
                    has_eos=True,
                    has_positional_emb=True,
                ),
            )
            self.num_layers = kwargs.pop("num_layers", 3)
            self.num_heads = kwargs.pop("num_heads", 1)
            self.scoring = kwargs.pop("scoring", "dot")
            self.in_drop_rates = kwargs.pop("in_drop_rates", (0.25, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.25)
            self.shortcut = kwargs.pop("shortcut", False)

        elif self.arch.lower() == "transformer":
            self.use_emb2init_hid = kwargs.pop("use_emb2init_hid", False)
            self.ff_dim = kwargs.pop("ff_dim", 256)
            self.embedding = kwargs.pop(
                "embedding",
                OneHotConfig(
                    tokens_key="trg_tokens",
                    field="text",
                    has_sos=True,
                    has_eos=True,
                    has_positional_emb=True,
                ),
            )
            self.num_layers = kwargs.pop("num_layers", 3)
            self.num_heads = kwargs.pop("num_heads", 8)
            self.scoring = kwargs.pop("scoring", "scaled_dot")
            self.in_drop_rates = kwargs.pop("in_drop_rates", (0.1, 0.0, 0.0))
            self.hid_drop_rate = kwargs.pop("hid_drop_rate", 0.1)
            self.shortcut = kwargs.pop("shortcut", False)

        else:
            raise ValueError(f"Invalid encoder architecture {self.arch}")

        # See Vaswani et al. (2017)
        self.weight_tying = kwargs.pop("weight_tying", False)
        self.weight_tying_scale = kwargs.pop(
            "weight_tying_scale", 1.0
        )  # self.emb_dim**-0.5
        self.teacher_forcing_rate = kwargs.pop("teacher_forcing_rate", 0.5)
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name_sep.join([self.scoring, self.arch, self.criterion])

    @property
    def ctx_dim(self):
        return self.in_dim

    @property
    def emb_dim(self):
        return self.embedding.emb_dim

    @property
    def full_hid_dim(self):
        if not self.shortcut:
            return self.hid_dim
        elif self.arch.lower() in ("lstm", "gru"):
            return self.hid_dim + self.emb_dim + self.ctx_dim
        else:
            return self.hid_dim + self.emb_dim

    @property
    def vocab(self):
        return self.embedding.vocab

    @property
    def max_len(self):
        return self.embedding.max_len

    def build_vocab(self, *partitions):
        flattened_partitions = [
            [
                {"trg_tokens": tokens}
                for entry in data
                for tokens in entry["full_trg_tokens"]
            ]
            for data in partitions
        ]
        self.embedding.build_vocab(*flattened_partitions)

    def instantiate(self):
        if self.arch.lower() in ("lstm", "gru"):
            return RNNGenerator(self)
        elif self.arch.lower() == "gehring":
            return GehringConvGenerator(self)
        elif self.arch.lower() == "transformer":
            return TransformerGenerator(self)


class Generator(DecoderBase, GeneratorMixin):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.vocab = config.vocab
        self.teacher_forcing_rate = config.teacher_forcing_rate
        self.shortcut = config.shortcut

        self.dropout = CombinedDropout(*config.in_drop_rates)
        self.embedding = config.embedding.instantiate()

        if config.weight_tying:
            assert config.full_hid_dim == config.emb_dim
            self.weight_tying_scale = config.weight_tying_scale
            self.hid2logit_bias = torch.nn.Parameter(torch.zeros(config.voc_dim))
        else:
            self.hid2logit = torch.nn.Linear(config.full_hid_dim, config.voc_dim)
            # TODO: Kaiming initialization contributes to faster convergence.
            # So, Kaiming initialization is suitable for pre-softmax layer? or for large weight matrix?
            reinit_layer_(self.hid2logit, "relu")

        # Every token in a batch should be assigned the same weight, so use `sum` as the reduction method.
        # This will not cause the model to generate shorter sequence, because the loss is summed over the ground-truth tokens,
        # which have fixed lengths.
        self.criterion = config.instantiate_criterion(
            ignore_index=config.pad_idx, reduction="sum"
        )

    def _forward_hid2logit(self, hidden: torch.Tensor):
        if hasattr(self, "hid2logit"):
            return self.hid2logit(hidden)
        else:
            return torch.nn.functional.linear(
                hidden,
                self.embedding.embedding.weight * self.weight_tying_scale,
                self.hid2logit_bias,
            )

    def _init_states(self, src_hidden: torch.Tensor, src_mask: torch.Tensor = None):
        raise NotImplementedError("Not Implemented `_init_states`")

    def forward_step(
        self,
        x_t: torch.Tensor,
        t: int,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        **states_utm1,
    ):
        raise NotImplementedError("Not Implemented `forward_step`")

    def forward2logits_step_by_step(
        self,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        return_atten_weight: bool = False,
    ):
        # Step by step forward
        # trg_tok_ids: (batch, trg_step)
        # src_hidden: (batch, src_step, ctx_dim)
        states_0 = self._init_states(src_hidden, src_mask=src_mask)

        logits, atten_weights = [], []
        # Prepare for step 1
        x_t = batch.trg_tok_ids[:, 0].unsqueeze(1)
        states_utm1 = states_0

        for t in range(1, batch.trg_tok_ids.size(1)):
            # t: 1, 2, ..., T-1
            # `utm1` means time step until `t-1`
            # `ut`   means time step until `t`
            logits_t, states_ut, atten_weight_t = self.forward_step(
                x_t, t, src_hidden, src_mask=src_mask, **states_utm1
            )

            logits.append(logits_t)
            atten_weights.append(atten_weight_t)
            top1 = logits_t.argmax(dim=-1)

            # Prepare for step t+1
            if self.training:
                tf_mask = (
                    torch.empty_like(top1).bernoulli(p=self.teacher_forcing_rate).bool()
                )
                x_t = torch.where(tf_mask, batch.trg_tok_ids[:, t].unsqueeze(1), top1)
            else:
                x_t = top1

            states_utm1 = states_ut

        # logits: (batch, trg_step-1, voc_dim)
        # atten_weights: (batch, trg_step-1, src_step)
        logits = torch.cat(logits, dim=1)
        atten_weights = torch.cat(atten_weights, dim=1)
        if return_atten_weight:
            return logits, atten_weights
        else:
            return logits

    def forward2logits_all_at_once(
        self,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        return_atten_weight: bool = False,
    ):
        raise NotImplementedError("Not Implemented `forward2logits_all_at_once`")

    def forward2logits(
        self,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        return_atten_weight: bool = False,
    ):
        # In training phase and with high teacher forcing rate, try at-once forward
        if self.training and self.teacher_forcing_rate > 0.99:
            try:
                return self.forward2logits_all_at_once(
                    batch,
                    src_hidden,
                    src_mask=src_mask,
                    return_atten_weight=return_atten_weight,
                )
            except NotImplementedError:
                pass
        # Fallback to step-by-step forward
        return self.forward2logits_step_by_step(
            batch,
            src_hidden,
            src_mask=src_mask,
            return_atten_weight=return_atten_weight,
        )

    def forward(
        self,
        batch: Batch,
        src_hidden: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        logits: torch.Tensor = None,
    ):
        # trg_tok_ids: (batch, trg_step)
        # logits: (batch, trg_step-1, voc_dim)
        # atten_weights: (batch, trg_step-1, src_step)
        if logits is None:
            assert src_hidden is not None
            logits = self.forward2logits(batch, src_hidden, src_mask)

        # `logits` accords with steps from 1 to `trg_step`
        # Positions of `pad_idx` will be ignored by `self.criterion`
        losses = [
            self.criterion(lg, t) for lg, t in zip(logits, batch.trg_tok_ids[:, 1:])
        ]
        # TODO: doubly stochastic regularization?
        return torch.stack(losses, dim=0)

    def decode(
        self,
        batch: Batch,
        src_hidden: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        logits: torch.Tensor = None,
    ):
        if logits is None:
            assert src_hidden is not None
            logits = self.forward2logits(batch, src_hidden, src_mask)

        batch_tok_ids = logits.argmax(dim=-1)
        batch_toks = []
        for tok_ids in batch_tok_ids.cpu().tolist():
            toks = [self.vocab.itos[tok_id] for tok_id in tok_ids]
            toks = list(itertools.takewhile(lambda tok: tok != "<eos>", toks))
            batch_toks.append(toks)
        return batch_toks

    def _expand_states(self, batch_size: int, **states):
        raise NotImplementedError("Not Implemented `_expand_states`")

    def _select_states(self, batch_indexing: torch.Tensor, **states):
        raise NotImplementedError("Not Implemented `_select_states`")

    def beam_search(
        self,
        beam_size: int,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
    ):
        batch_best_trg_toks = []

        for k in range(src_hidden.size(0)):
            # The `k`-th source states
            curr_src_hidden = src_hidden[k].unsqueeze(0)
            curr_src_mask = None if src_mask is None else src_mask[k].unsqueeze(0)

            states_0 = self._init_states(curr_src_hidden, src_mask=curr_src_mask)
            # x_t: (batch=1, step=1)
            x_t = batch.trg_tok_ids[k, 0].view(1, 1)
            states_utm1 = states_0

            beam_trg_toks = [[]]
            ended_beam_trg_toks, ended_log_scores = [], []

            for t in range(1, batch.trg_tok_ids.size(1)):
                # t: 1, 2, ..., T-1
                # logits_t: (batch=1, step=1, voc_dim)
                logits_t, states_ut, _ = self.forward_step(
                    x_t,
                    t,
                    src_hidden=curr_src_hidden.expand(x_t.size(0), -1, -1),
                    src_mask=None
                    if src_mask is None
                    else curr_src_mask.expand(x_t.size(0), -1),
                    **states_utm1,
                )

                if t == 1:
                    # log_scores: (batch=1, step=1, voc_dim)
                    log_scores = logits_t.log_softmax(dim=-1)
                    # log_scores/topk_indexes: (batch=beam, step=1)
                    log_scores, topk_indexes = (
                        log_scores.permute(0, 2, 1)
                        .flatten(end_dim=1)
                        .topk(beam_size, dim=0)
                    )

                    states_ut = self._expand_states(beam_size, **states_ut)
                else:
                    # log_scores: (batch=beam, step=1, voc_dim)
                    log_scores = log_scores.unsqueeze(-1) + logits_t.log_softmax(dim=-1)
                    # log_scores/topk_indexes: (batch=beam, step=1)
                    log_scores, topk_indexes = (
                        log_scores.permute(0, 2, 1)
                        .flatten(end_dim=1)
                        .topk(x_t.size(0), dim=0)
                    )

                # Treat it as having a batch size of `beam_size`
                # prev_indexes/x_t: (batch=beam, step=1)
                prev_indexes, x_t = (topk_indexes // self.voc_dim), (
                    topk_indexes % self.voc_dim
                )

                beam_trg_toks = [
                    beam_trg_toks[prev_idx] + [self.vocab.itos[tok_id]]
                    for prev_idx, tok_id in zip(
                        prev_indexes.flatten().cpu().tolist(),
                        x_t.flatten().cpu().tolist(),
                    )
                ]

                # is_end: (batch=beam, )
                is_end = x_t.flatten() == self.eos_idx

                if is_end.any().item():
                    # Remove `<eos>`
                    ended_beam_trg_toks.extend(
                        [
                            trg_toks[:-1]
                            for trg_toks, ise in zip(
                                beam_trg_toks, is_end.cpu().tolist()
                            )
                            if ise
                        ]
                    )
                    ended_log_scores.extend(log_scores[is_end].flatten().cpu().tolist())

                    if len(ended_beam_trg_toks) == beam_size:
                        break

                    beam_trg_toks = [
                        trg_toks
                        for trg_toks, ise in zip(beam_trg_toks, is_end.cpu().tolist())
                        if not ise
                    ]
                    x_t = x_t[~is_end]
                    states_ut = self._select_states(~is_end, **states_ut)
                    log_scores = log_scores[~is_end]

                states_utm1 = states_ut

            # Use *ended* sequences only, unless all sequences are not *ended*
            if len(ended_beam_trg_toks) == 0:
                ended_beam_trg_toks = beam_trg_toks
                ended_log_scores = log_scores.flatten().cpu().tolist()

            max_log_score, best_trg_toks = max(
                zip(ended_log_scores, ended_beam_trg_toks)
            )
            batch_best_trg_toks.append(best_trg_toks)

        return batch_best_trg_toks


class RNNGenerator(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)

        rnn_config = {
            "input_size": config.ctx_dim + config.emb_dim,
            "hidden_size": config.hid_dim,
            "num_layers": config.num_layers,
            "batch_first": True,
            "bidirectional": False,
            "dropout": 0.0 if config.num_layers <= 1 else config.hid_drop_rate,
        }

        if config.arch.lower() == "lstm":
            self.rnn = torch.nn.LSTM(**rnn_config)
            reinit_lstm_(self.rnn)
        elif config.arch.lower() == "gru":
            self.rnn = torch.nn.GRU(**rnn_config)
            reinit_gru_(self.rnn)

        if config.init_ctx_mode.lower().endswith(
            "_pooling"
        ) or config.init_ctx_mode.lower().startswith("rnn_last"):
            self.init_ctx = SequencePooling(
                mode=config.init_ctx_mode.replace("_pooling", "")
            )
        else:
            self.init_query = torch.nn.Parameter(torch.empty(config.hid_dim))
            reinit_vector_parameter_(self.init_query)

        self.ctx2h_0 = torch.nn.Linear(
            config.ctx_dim, config.hid_dim * config.num_layers
        )
        reinit_layer_(self.ctx2h_0, "tanh")
        if isinstance(self.rnn, torch.nn.LSTM):
            self.ctx2c_0 = torch.nn.Linear(
                config.ctx_dim, config.hid_dim * config.num_layers
            )
            reinit_layer_(self.ctx2c_0, "tanh")
        # RNN hidden states are tanh-activated
        self.tanh = torch.nn.Tanh()
        self.attention = SequenceAttention(
            key_dim=config.ctx_dim,
            query_dim=config.hid_dim,
            num_heads=config.num_heads,
            scoring=config.scoring,
            external_query=True,
        )

    def _init_states(self, src_hidden: torch.Tensor, src_mask: torch.Tensor = None):
        if hasattr(self, "init_ctx"):
            context_0 = self.init_ctx(src_hidden, mask=src_mask)
        else:
            context_0 = self.attention(src_hidden, mask=src_mask, query=self.init_query)

        h_0 = self.tanh(self.ctx2h_0(context_0))
        h_0 = (
            h_0.view(-1, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        if isinstance(self.rnn, torch.nn.LSTM):
            c_0 = self.tanh(self.ctx2c_0(context_0))
            c_0 = (
                c_0.view(-1, self.rnn.num_layers, self.rnn.hidden_size)
                .permute(1, 0, 2)
                .contiguous()
            )
            h_0 = (h_0, c_0)
        return {"h_tm1": h_0}

    def _get_top_hidden(self, h_t: torch.Tensor):
        if isinstance(self.rnn, torch.nn.LSTM):
            # h_t: (h_t, c_t)
            return h_t[0][-1].unsqueeze(1)
        else:
            # h_t: (num_layers, batch, hid_dim) -> (batch, step=1, hid_dim)
            return h_t[-1].unsqueeze(1)

    def _expand_states(self, batch_size: int, h_tm1: torch.Tensor):
        if isinstance(self.rnn, torch.nn.LSTM):
            h_tm1 = (
                h_tm1[0].expand(-1, batch_size, -1).contiguous(),
                h_tm1[1].expand(-1, batch_size, -1).contiguous(),
            )
        else:
            h_tm1 = h_tm1.expand(-1, batch_size, -1).contiguous()
        return {"h_tm1": h_tm1}

    def _select_states(self, batch_indexing: torch.Tensor, h_tm1: torch.Tensor):
        if isinstance(self.rnn, torch.nn.LSTM):
            h_tm1 = (
                h_tm1[0][:, batch_indexing].contiguous(),
                h_tm1[1][:, batch_indexing].contiguous(),
            )
        else:
            h_tm1 = h_tm1[:, batch_indexing].contiguous()
        return {"h_tm1": h_tm1}

    def forward_step(
        self,
        x_t: torch.Tensor,
        t: int,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        h_tm1: torch.Tensor = None,
    ):
        # x_t: (batch, step=1)
        # src_hidden: (batch, src_step, ctx_dim)
        # src_mask: (batch, src_step)
        # h_tm1/h_t: (num_layers, batch, hid_dim)

        # embedded_t: (batch, step=1, emb_dim)
        embedded_t = self.dropout(self.embedding(x_t, start_position_id=t - 1))

        # query_t: (batch, step=1, hid_dim)
        # context_t: (batch, step=1, ctx_dim)
        context_t, atten_weight_t = self.attention(
            src_hidden,
            mask=src_mask,
            query=self._get_top_hidden(h_tm1),
            return_atten_weight=True,
        )

        # Forward one RNN step
        _, h_t = self.rnn(torch.cat([embedded_t, context_t], dim=-1), h_tm1)

        hidden_t = self._get_top_hidden(h_t)
        if self.shortcut:
            hidden_t = torch.cat([hidden_t, embedded_t, context_t], dim=-1)

        # logits_t: (batch, step=1, voc_dim)
        logits_t = self._forward_hid2logit(hidden_t)
        return logits_t, {"h_tm1": h_t}, atten_weight_t


class GehringConvGenerator(Generator):
    """Convolutional sequence decoder by Gehring et al. (2017).

    References
    ----------
    Gehring, J., et al. 2017. Convolutional Sequence to Sequence Learning.
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        self.emb2init_hid = torch.nn.Linear(config.emb_dim, config.hid_dim * 2)
        self.glu = torch.nn.GLU(dim=-1)
        reinit_layer_(self.emb2init_hid, "glu")

        self.conv_blocks = torch.nn.ModuleList(
            [
                ConvBlock(
                    in_dim=config.hid_dim,
                    out_dim=config.hid_dim,
                    kernel_size=config.kernel_size,
                    drop_rate=config.hid_drop_rate,  # Note to apply dropout to `init_hidden`
                    padding_mode="none",  # The paddings have to be manually operated
                    nonlinearity="glu",
                )
                for k in range(config.num_layers)
            ]
        )
        self.register_buffer(
            "_pre_padding", torch.zeros(config.hid_dim, config.kernel_size - 1)
        )

        # Maybe always use **affined** hidden states for attention computation branches?
        self.pre_atten_affine = torch.nn.Linear(
            config.hid_dim, config.ctx_dim
        )  # Use conv-layer of kernel size 1
        reinit_layer_(self.pre_atten_affine, "linear")
        self.attention = SequenceAttention(
            key_dim=config.hid_dim,
            query_dim=config.hid_dim,
            num_heads=config.num_heads,
            scoring=config.scoring,
            external_query=True,
        )
        self.post_atten_affine = torch.nn.Linear(config.ctx_dim, config.hid_dim)
        reinit_layer_(self.post_atten_affine, "linear")
        self.scale = config.scale

    @property
    def kernel_size(self):
        return self.conv_blocks[0].kernel_size

    def _init_states(self, src_hidden: torch.Tensor, src_mask: torch.Tensor = None):
        return {
            "hidden_stack_utm1": [
                self._pre_padding.expand(src_hidden.size(0), -1, -1)
                for _ in self.conv_blocks
            ]
        }

    def _expand_states(self, batch_size: int, hidden_stack_utm1: List[torch.Tensor]):
        return {
            "hidden_stack_utm1": [
                hidden_utm1.expand(batch_size, -1, -1)
                for hidden_utm1 in hidden_stack_utm1
            ]
        }

    def _select_states(
        self, batch_indexing: torch.Tensor, hidden_stack_utm1: List[torch.Tensor]
    ):
        return {
            "hidden_stack_utm1": [
                hidden_utm1[batch_indexing] for hidden_utm1 in hidden_stack_utm1
            ]
        }

    def forward_step(
        self,
        x_t: torch.Tensor,
        t: int,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        hidden_stack_utm1: List[torch.Tensor] = None,
    ):
        # x_t: (batch, step=1)
        # embedded_t: (batch, step=1, emb_dim)
        embedded_t = self.dropout(self.embedding(x_t, start_position_id=t - 1))

        # init_hidden_t: (batch, step=1, hid_dim)
        init_hidden_t = self.glu(self.emb2init_hid(embedded_t))

        # hidden_t: (batch, hid_dim/channels, step=1)
        hidden_t = init_hidden_t.permute(0, 2, 1)

        hidden_stack_ut = []
        for conv_block, hidden_utm1 in zip(self.conv_blocks, hidden_stack_utm1):
            # `utm1` means time step until `t-1`
            # `ut`   means time step until `t`
            hidden_ut = torch.cat([hidden_utm1, hidden_t], dim=-1)
            hidden_stack_ut.append(hidden_ut)

            # conved_t: (batch, hid_dim/channels, step=1)
            conved_t = conv_block(hidden_ut[:, :, -self.kernel_size :])
            init_hidden_conved_t = self.pre_atten_affine(
                (init_hidden_t + conved_t.permute(0, 2, 1)) * self.scale
            )

            # context_t: (batch, step=1, ctx_dim=hid_dim)
            context_t, atten_weight_t = self.attention(
                src_hidden,
                mask=src_mask,
                query=init_hidden_conved_t,
                return_atten_weight=True,
            )
            context_t = self.post_atten_affine(context_t)

            # conved_context_t: (batch, hid_dim/channels, step=1)
            conved_context_t = (conved_t + context_t.permute(0, 2, 1)) * self.scale
            hidden_t = (hidden_t + conved_context_t) * self.scale

        # final_hidden_t: (batch, step=1, hid_dim)
        final_hidden_t = hidden_t.permute(0, 2, 1)
        if self.shortcut:
            final_hidden_t = torch.cat([final_hidden_t, embedded_t], dim=-1)

        # logits_t: (batch, step=1, voc_dim)
        logits_t = self._forward_hid2logit(final_hidden_t)
        return logits_t, {"hidden_stack_utm1": hidden_stack_ut}, atten_weight_t

    def forward2logits_all_at_once(
        self,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        return_atten_weight: bool = False,
    ):
        states_ut0 = self._init_states(src_hidden, src_mask=src_mask)
        hidden_stack_ut0 = states_ut0["hidden_stack_utm1"]

        # embedded: (batch, step, emb_dim); step = trg_step-1
        embedded = self.dropout(self.embedding(batch.trg_tok_ids[:, :-1]))

        # init_hidden: (batch, step, hid_dim)
        init_hidden = self.glu(self.emb2init_hid(embedded))

        # hidden: (batch, hid_dim, step)
        hidden = init_hidden.permute(0, 2, 1)

        for conv_block, hidden_ut0 in zip(self.conv_blocks, hidden_stack_ut0):
            hidden_padded = torch.cat([hidden_ut0, hidden], dim=-1)

            # conved: (batch, hid_dim/channels, step)
            conved = conv_block(
                hidden_padded
            )  # No need for mask, since padding are all in front
            # TODO: add `hidden` or `init_hidden`? add before or after `pre_atten_affine`?
            init_hidden_conved = self.pre_atten_affine(
                (init_hidden + conved.permute(0, 2, 1)) * self.scale
            )

            # context: (batch, step, ctx_dim=hid_dim)
            context, atten_weight = self.attention(
                src_hidden,
                mask=src_mask,
                query=init_hidden_conved,
                return_atten_weight=True,
            )
            context = self.post_atten_affine(context)

            # conved_context: (batch, hid_dim/channels, step)
            conved_context = (conved + context.permute(0, 2, 1)) * self.scale
            hidden = (hidden + conved_context) * self.scale

        # final_hidden: (batch, step, hid_dim)
        final_hidden = hidden.permute(0, 2, 1)
        if self.shortcut:
            final_hidden = torch.cat([final_hidden, embedded], dim=-1)

        # logits: (batch, step, voc_dim)
        logits = self._forward_hid2logit(final_hidden)
        if return_atten_weight:
            return logits, atten_weight  # Only atten_weight on the top layer
        else:
            return logits


class TransformerGenerator(Generator):
    """Transformer decoder by Vaswani et al. (2017).

    References
    ----------
    Vaswani, A., et al. 2017. Attention is All You Need.
    """

    def __init__(self, config: GeneratorConfig):
        super().__init__(config)
        if config.use_emb2init_hid:
            self.emb2init_hid = torch.nn.Linear(config.emb_dim, config.hid_dim)
            self.relu = torch.nn.ReLU()
            reinit_layer_(self.emb2init_hid, "relu")
        else:
            assert config.hid_dim == config.emb_dim

        self.tf_blocks = torch.nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hid_dim=config.hid_dim,
                    ff_dim=config.ff_dim,
                    ctx_dim=config.ctx_dim,
                    num_heads=config.num_heads,
                    drop_rate=(
                        0.0
                        if (k == 0 and not config.use_emb2init_hid)
                        else config.hid_drop_rate
                    ),
                    nonlinearity="relu",
                )
                for k in range(config.num_layers)
            ]
        )
        # Actually no pre-padding is needed
        # This is a placeholder for `_init_states`, and thus for consistent/easier implementation of `forward_step`
        self.register_buffer("_pre_padding", torch.zeros(0, config.hid_dim))

    def _init_states(self, src_hidden: torch.Tensor, src_mask: torch.Tensor = None):
        return {
            "hidden_stack_utm1": [
                self._pre_padding.expand(src_hidden.size(0), -1, -1)
                for _ in self.tf_blocks
            ]
        }

    def _expand_states(self, batch_size: int, hidden_stack_utm1: List[torch.Tensor]):
        return {
            "hidden_stack_utm1": [
                hidden_utm1.expand(batch_size, -1, -1)
                for hidden_utm1 in hidden_stack_utm1
            ]
        }

    def _select_states(
        self, batch_indexing: torch.Tensor, hidden_stack_utm1: List[torch.Tensor]
    ):
        return {
            "hidden_stack_utm1": [
                hidden_utm1[batch_indexing] for hidden_utm1 in hidden_stack_utm1
            ]
        }

    def forward_step(
        self,
        x_t: torch.Tensor,
        t: int,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        hidden_stack_utm1: List[torch.Tensor] = None,
    ):
        # x_t: (batch, step=1)
        # embedded_t: (batch, step=1, emb_dim)
        embedded_t = self.dropout(self.embedding(x_t, start_position_id=t - 1))

        if hasattr(self, "emb2init_hid"):
            hidden_t = self.relu(self.emb2init_hid(embedded_t))
        else:
            hidden_t = embedded_t

        hidden_stack_ut = []
        for tf_block, hidden_utm1 in zip(self.tf_blocks, hidden_stack_utm1):
            hidden_ut = torch.cat([hidden_utm1, hidden_t], dim=1)
            hidden_stack_ut.append(hidden_ut)

            hidden_t, atten_weight_t, cross_atten_weight_t = tf_block(
                hidden_ut,
                src_hidden,
                src_mask=src_mask,
                last_step=True,
                return_atten_weight=True,
            )

        # hidden_t: (batch, step=1, hid_dim)
        if self.shortcut:
            hidden_t = torch.cat([hidden_t, embedded_t], dim=-1)

        # logits_t: (batch, step=1, voc_dim)
        logits_t = self._forward_hid2logit(hidden_t)
        return logits_t, {"hidden_stack_utm1": hidden_stack_ut}, cross_atten_weight_t

    def forward2logits_all_at_once(
        self,
        batch: Batch,
        src_hidden: torch.Tensor,
        src_mask: torch.Tensor = None,
        return_atten_weight: bool = False,
    ):
        # embedded: (batch, step, emb_dim); step = trg_step-1
        embedded = self.dropout(self.embedding(batch.trg_tok_ids[:, :-1]))

        if hasattr(self, "emb2init_hid"):
            hidden = self.relu(self.emb2init_hid(embedded))
        else:
            hidden = embedded

        for tf_block in self.tf_blocks:
            hidden, atten_weight, cross_atten_weight = tf_block(
                hidden,
                src_hidden,
                src_mask=src_mask,
                last_step=False,
                return_atten_weight=True,
            )

        if self.shortcut:
            hidden = torch.cat([hidden, embedded], dim=-1)

        # logits: (batch, step, voc_dim)
        logits = self._forward_hid2logit(hidden)
        if return_atten_weight:
            return logits, cross_atten_weight  # Only atten_weight on the top layer
        else:
            return logits
