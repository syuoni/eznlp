# -*- coding: utf-8 -*-
from .embedding import SinusoidPositionalEncoding
from .aggregation import SequencePooling, SequenceGroupAggregating, ScalarMix
from .attention import SequenceAttention
from .block import FeedForwardBlock, ConvBlock, MultiheadAttention, TransformerEncoderBlock, TransformerDecoderBlock
from .dropout import WordDropout, LockedDropout, CombinedDropout
from .query_bert_like import QueryBertLikeEncoder
from .crf import CRF
from .affine_fusor import MultiAffineFusor, BiAffineFusor, TriAffineFusor, QuadAffineFusor
from .loss import SoftLabelCrossEntropyLoss, SmoothLabelCrossEntropyLoss, FocalLoss
from .distance import MultiKernelMaxMeanDiscrepancyLoss

__all__ = [
    'SinusoidPositionalEncoding', 'SequencePooling', 'SequenceGroupAggregating',
    'ScalarMix', 'SequenceAttention', 'FeedForwardBlock', 'ConvBlock',
    'MultiheadAttention', 'TransformerEncoderBlock', 'TransformerDecoderBlock',
    'WordDropout', 'LockedDropout', 'CombinedDropout', 'QueryBertLikeEncoder',
    'CRF', 'MultiAffineFusor', 'BiAffineFusor', 'TriAffineFusor', 'QuadAffineFusor',
    'SoftLabelCrossEntropyLoss', 'SmoothLabelCrossEntropyLoss', 'FocalLoss',
    'MultiKernelMaxMeanDiscrepancyLoss'
]
