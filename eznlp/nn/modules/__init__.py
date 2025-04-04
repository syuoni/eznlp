# -*- coding: utf-8 -*-
from .affine_fusor import (
    BiAffineFusor,
    MultiAffineFusor,
    QuadAffineFusor,
    TriAffineFusor,
)
from .aggregation import ScalarMix, SequenceGroupAggregating, SequencePooling
from .attention import SequenceAttention
from .block import (
    ConvBlock,
    FeedForwardBlock,
    MultiheadAttention,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)
from .crf import CRF
from .distance import MultiKernelMaxMeanDiscrepancyLoss
from .dropout import CombinedDropout, LockedDropout, WordDropout
from .embedding import SinusoidPositionalEncoding
from .loss import FocalLoss, SmoothLabelCrossEntropyLoss, SoftLabelCrossEntropyLoss
from .query_bert_like import QueryBertLikeEncoder

__all__ = [
    "SinusoidPositionalEncoding",
    "SequencePooling",
    "SequenceGroupAggregating",
    "ScalarMix",
    "SequenceAttention",
    "FeedForwardBlock",
    "ConvBlock",
    "MultiheadAttention",
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "WordDropout",
    "LockedDropout",
    "CombinedDropout",
    "QueryBertLikeEncoder",
    "CRF",
    "MultiAffineFusor",
    "BiAffineFusor",
    "TriAffineFusor",
    "QuadAffineFusor",
    "SoftLabelCrossEntropyLoss",
    "SmoothLabelCrossEntropyLoss",
    "FocalLoss",
    "MultiKernelMaxMeanDiscrepancyLoss",
]
