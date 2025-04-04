# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import List

import torch
import transformers

from ..config import Config
from ..nn.init import reinit_embedding_, reinit_vector_parameter_
from ..nn.modules import QueryBertLikeEncoder, SequenceAttention, SequencePooling


class MaskedSpanBertLikeConfig(Config):
    def __init__(self, **kwargs):
        self.bert_like: transformers.PreTrainedModel = kwargs.pop("bert_like")
        self.hid_dim = self.bert_like.config.hidden_size

        self.arch = kwargs.pop("arch", "BERT")
        self.freeze = kwargs.pop("freeze", True)

        self.num_layers = kwargs.pop("num_layers", None)
        if self.num_layers is None:
            self.num_layers = self.bert_like.config.num_hidden_layers
        assert 0 < self.num_layers <= self.bert_like.config.num_hidden_layers

        self.use_init_size_emb = kwargs.pop("use_init_size_emb", False)
        self.use_init_dist_emb = kwargs.pop("use_init_dist_emb", False)
        self.share_weights_ext = kwargs.pop(
            "share_weights_ext", True
        )  # Share weights externally, i.e., with `bert_like`
        self.share_weights_int = kwargs.pop(
            "share_weights_int", True
        )  # Share weights internally, i.e., between `query_bert_like` of different span sizes
        assert self.share_weights_int
        self.init_agg_mode = kwargs.pop("init_agg_mode", "max_pooling")
        self.init_drop_rate = kwargs.pop("init_drop_rate", 0.2)
        super().__init__(**kwargs)

    @property
    def name(self):
        return f"{self.arch}({self.num_layers})"

    @property
    def out_dim(self):
        return self.hid_dim

    def __getstate__(self):
        state = self.__dict__.copy()
        state["bert_like"] = None
        return state

    def instantiate(self):
        return MaskedSpanBertLikeEncoder(self)


class MaskedSpanBertLikeEncoder(torch.nn.Module):
    def __init__(self, config: MaskedSpanBertLikeConfig):
        super().__init__()
        if config.init_agg_mode.lower().endswith("_pooling"):
            self.init_aggregating = SequencePooling(
                mode=config.init_agg_mode.replace("_pooling", "")
            )
        elif config.init_agg_mode.lower().endswith("_attention"):
            self.init_aggregating = SequenceAttention(
                config.hid_dim, scoring=config.init_agg_mode.replace("_attention", "")
            )

        self.dropout = torch.nn.Dropout(config.init_drop_rate)

        if config.use_init_size_emb:
            self.size_embedding = torch.nn.Embedding(
                config.max_size_id + 1, config.hid_dim
            )
            reinit_embedding_(self.size_embedding)

        if config.use_init_dist_emb:
            self.dist_embedding = torch.nn.Embedding(
                config.max_dist_id + 1, config.hid_dim
            )
            reinit_embedding_(self.dist_embedding)

        assert config.share_weights_int
        # Share the module across all span sizes
        self.query_bert_like = QueryBertLikeEncoder(
            config.bert_like.encoder,
            num_layers=config.num_layers,
            share_weights=config.share_weights_ext,
        )
        # Trainable context vector for the span covering the whole sequence
        self.zero_context = torch.nn.Parameter(torch.empty(config.hid_dim))
        reinit_vector_parameter_(self.zero_context)

        self.freeze = config.freeze
        self.num_layers = config.num_layers
        self.share_weights_ext = config.share_weights_ext
        self.share_weights_int = config.share_weights_int

    @property
    def freeze(self):
        return self._freeze

    @freeze.setter
    def freeze(self, freeze: bool):
        self._freeze = freeze
        self.query_bert_like.requires_grad_(not freeze)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor):
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.size()})"
            )
        return attention_mask.float() * -10000

    def _forward_aggregation(
        self,
        all_hidden_states: List[torch.Tensor],
        init_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        init_embedded: torch.Tensor = None,
    ):
        batch_size, num_steps, hid_dim = all_hidden_states[0].size()
        num_items = init_mask.size(1)
        if num_items == 0:
            return dict(
                last_query_state=torch.empty(
                    batch_size, 0, hid_dim, device=init_mask.device
                )
            )

        # All masked attention will produce NaN gradients
        zero_indic = init_mask.all(dim=-1).logical_or(attention_mask.all(dim=-1))
        if zero_indic.any().item():
            # DO NOT inplace modify tensors
            # Note that this function may be called for multiple times, and these masking tensors may be reused
            init_mask = init_mask.masked_fill(zero_indic.unsqueeze(-1), False)
            attention_mask = attention_mask.masked_fill(zero_indic.unsqueeze(-1), False)

        if isinstance(self.init_aggregating, (SequencePooling, SequenceAttention)):
            # reshaped_states: (B, L, H) -> (B, NI, L, H) -> (B*NI, L, H)
            reshaped_states0 = (
                all_hidden_states[0]
                .unsqueeze(1)
                .expand(-1, num_items, -1, -1)
                .contiguous()
                .view(-1, num_steps, hid_dim)
            )

            # init_mask: (B, NI, L) -> (B*NI, L)
            # query_states: (B*NI, H) -> (B, NI, H)
            # Initial context queries are also created from span attention mask (instead of context attention mask)
            query_states = self.init_aggregating(
                self.dropout(reshaped_states0), mask=init_mask.view(-1, num_steps)
            )
            query_states = query_states.view(batch_size, -1, hid_dim)

        if init_embedded is not None:
            query_states = query_states + self.dropout(init_embedded)

        # attention_mask: (B, NI, L)
        query_outs = self.query_bert_like(
            query_states,
            all_hidden_states,
            self.get_extended_attention_mask(attention_mask),
        )

        if zero_indic.any().item():
            # DO NOT inplace modify tensors
            query_outs["last_query_state"] = torch.where(
                zero_indic.unsqueeze(-1),
                self.zero_context,
                query_outs["last_query_state"],
            )

        return query_outs

    def forward(
        self,
        all_hidden_states: List[torch.Tensor],
        span_size_ids: torch.Tensor,
        ck2tok_mask: torch.Tensor,
        cp_dist_ids: torch.Tensor = None,
        ctx2tok_mask: torch.Tensor = None,
        pair2tok_mask: torch.Tensor = None,
    ):
        # Remove the unused layers of hidden states
        all_hidden_states = all_hidden_states[-(self.num_layers + 1) :]

        all_last_query_states = OrderedDict()
        span_init_embedded = None
        if hasattr(self, "size_embedding"):
            # size_embedded: (batch, num_chunks, hid_dim)
            size_embedded = self.size_embedding(span_size_ids)
            span_init_embedded = size_embedded
        span_query_outs = self._forward_aggregation(
            all_hidden_states,
            init_mask=ck2tok_mask,
            attention_mask=ck2tok_mask,
            init_embedded=span_init_embedded,
        )
        all_last_query_states["span_query_state"] = span_query_outs["last_query_state"]

        if ctx2tok_mask is not None:
            if pair2tok_mask is None:
                ctx_query_outs = self._forward_aggregation(
                    all_hidden_states,
                    init_mask=ck2tok_mask,
                    attention_mask=ctx2tok_mask,
                    init_embedded=span_init_embedded,
                )
            else:
                ctx_init_embedded = None
                if hasattr(self, "dist_embedding"):
                    # dist_embedded: (batch, num_pairs, hid_dim)
                    dist_embedded = self.dist_embedding(cp_dist_ids)
                    ctx_init_embedded = dist_embedded
                ctx_query_outs = self._forward_aggregation(
                    all_hidden_states,
                    init_mask=pair2tok_mask,
                    attention_mask=ctx2tok_mask,
                    init_embedded=ctx_init_embedded,
                )
            all_last_query_states["ctx_query_state"] = ctx_query_outs[
                "last_query_state"
            ]

        return all_last_query_states
